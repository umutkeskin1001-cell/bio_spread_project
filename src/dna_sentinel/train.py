from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.swa_utils import SWALR, AveragedModel

from dna_sentinel.model import Cassiopeia, _focal_bce
from dna_sentinel.utils import WindowDropout, binary_metrics, logger, multiclass_metrics, set_seed


def _inverse_label_frequency(labels: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    y = labels.detach().cpu().long().view(-1)
    mx = int(y.max().item()) + 1 if y.numel() else 1
    counts = torch.bincount(y, minlength=mx).float()
    counts = counts.clamp_min(1.0)
    return (1.0 / (counts + eps))[y]


def _balanced_sample_weights(data: dict) -> torch.Tensor:
    return (
        _inverse_label_frequency(data["mobility"])
        + _inverse_label_frequency(data["amr"].long())
        + _inverse_label_frequency(data["expansion"].long())
    ) / 3.0


def _epoch_indices(
    n_train: int, data: dict, config: dict, generator: torch.Generator | None = None,
    cached_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if config.get("balanced_sampling", False):
        w = cached_weights if cached_weights is not None else _balanced_sample_weights(data)
        return torch.multinomial(w / w.mean().clamp_min(1e-6), n_train, replacement=True, generator=generator)
    return torch.randperm(n_train, generator=generator)


def _selection_score(metrics: dict[str, float], config: dict) -> float:
    mob = float(metrics.get("mobility_balanced_accuracy", 0.0))
    amr = float(metrics.get("amr_auroc", 0.0))
    exp = float(metrics.get("expansion_auroc", 0.0))
    mode = config.get("score_mode", "equal")
    if mode == "legacy":
        return amr + exp + 2.0 * mob
    if mode != "equal":
        raise ValueError(f"unknown score_mode: {mode}")
    base = (mob + amr + exp) / 3.0
    ece_penalty = float(metrics.get("ece_penalty", 0.0))
    drift_penalty = float(metrics.get("drift_penalty", 0.0))
    return base - ece_penalty - drift_penalty


def _build_optimizer(model, config):
    backbone_lr = config.get("backbone_lr", config.get("lr", 1e-3))
    head_lr = config.get("head_lr", config.get("lr", 1e-3))
    backbone, heads = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (heads if n.startswith(("mob_", "amr_", "exp_", "log_vars")) else backbone).append(p)
    groups = [{"params": g, "lr": lr} for g, lr in [(backbone, backbone_lr), (heads, head_lr)] if g]
    return torch.optim.AdamW(groups, lr=backbone_lr, weight_decay=config.get("weight_decay", 0.05))


def _build_scheduler(optimizer, config):
    epochs = int(config["epochs"])
    warmup = int(config.get("warmup_epochs", 0))
    min_lr = float(config.get("min_lr", 1e-5))
    if warmup > 0 and epochs > warmup:
        return SequentialLR(
            optimizer,
            [
                LinearLR(optimizer, start_factor=0.1, total_iters=warmup),
                CosineAnnealingLR(optimizer, T_max=epochs - warmup, eta_min=min_lr),
            ],
            milestones=[warmup],
        )
    return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)


def _consistency_loss(out_ref: dict, out_aug: dict, temperature: float = 1.0, expansion_classes: int = 1) -> torch.Tensor:
    t = max(float(temperature), 1e-6)
    mob_ref = torch.softmax(out_ref["mobility_logits"].detach() / t, dim=-1)
    mob = F.kl_div(torch.log_softmax(out_aug["mobility_logits"] / t, dim=-1), mob_ref, reduction="batchmean") * (t * t)
    amr_ref = torch.sigmoid(out_ref["amr_logits"].detach())
    amr_loss = F.mse_loss(torch.sigmoid(out_aug["amr_logits"]), amr_ref)
    if expansion_classes > 1:
        exp_ref = torch.softmax(out_ref["expansion_logits"].detach() / t, dim=-1)
        exp_loss = (
            F.kl_div(
                torch.log_softmax(out_aug["expansion_logits"] / t, dim=-1),
                exp_ref, reduction="batchmean"
            ) * (t * t)
        )
    else:
        exp_ref = torch.sigmoid(out_ref["expansion_logits"].detach())
        exp_loss = F.mse_loss(torch.sigmoid(out_aug["expansion_logits"]), exp_ref)
    return mob + amr_loss + exp_loss


@torch.inference_mode()
def evaluate(model, data, device="cpu", batch_size=128, return_probs=False):
    model.eval()
    n = len(data["features"])
    sf = data.get("struct_features", None)
    sc = data.get("scale_ids", None)
    dev = device if isinstance(device, torch.device) else torch.device(device)
    non_blocking = dev.type == "cuda"
    mob_p, amr_p, exp_p = [], [], []
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        out = model(
            data["features"][s:e].to(dev, non_blocking=non_blocking),
            data["masks"][s:e].to(dev, non_blocking=non_blocking),
            struct_features=sf[s:e].to(dev, non_blocking=non_blocking) if sf is not None else None,
            scale_ids=sc[s:e].to(dev, non_blocking=non_blocking) if sc is not None else None,
        )
        mob_p.append(torch.softmax(out["mobility_logits"], dim=-1).cpu())
        amr_p.append(torch.sigmoid(out["amr_logits"]).cpu())
        exp = (
            torch.softmax(out["expansion_logits"], dim=-1).cpu()
            if model.config.expansion_classes > 1
            else torch.sigmoid(out["expansion_logits"]).cpu()
        )
        exp_p.append(exp)
    mob_np = torch.cat(mob_p).numpy()
    amr_np = torch.cat(amr_p).numpy()
    exp_np = torch.cat(exp_p).numpy()
    amr_true = data["amr"].cpu().numpy()
    exp_true = data["expansion"].cpu().numpy()
    m = multiclass_metrics(data["mobility"].cpu().numpy(), mob_np, "mobility")
    if model.config.amr_classes > 1 and amr_true.ndim > 1:
        aurocs = [
            binary_metrics(amr_true[:, i], amr_np[:, i], f"amr_{i}")[f"amr_{i}_auroc"]
            for i in range(min(model.config.amr_classes, amr_true.shape[1]))
            if len(np.unique(amr_true[:, i])) >= 2
        ]
        m["amr_auroc"] = float(np.mean(aurocs)) if aurocs else 0.5
    else:
        m.update(binary_metrics(amr_true, amr_np, "amr"))
    if model.config.expansion_classes > 1:
        n_cls = min(model.config.expansion_classes, exp_np.shape[1])
        aurocs = [
            roc_auc_score((exp_true == i).astype(float), exp_np[:, i])
            for i in range(n_cls)
            if len(np.unique(exp_true == i)) >= 2 and not np.isnan(exp_np[:, i]).any()
        ]
        m["expansion_auroc"] = float(np.mean(aurocs)) if aurocs else 0.5
        m["expansion_accuracy"] = float(np.mean(exp_true == exp_np.argmax(axis=1)))
    else:
        m.update(binary_metrics(exp_true, exp_np, "expansion"))
    return (m, mob_np, amr_np, exp_np) if return_probs else m


def _compute_batch_loss(model, out, mob_t, amr_t, exp_t, amr_pw, exp_pw, exp_pw_mc, gamma):
    return model.compute_loss(
        out["mobility_logits"], out["amr_logits"], out["expansion_logits"],
        mob_t.long(), amr_t, exp_t, amr_pw, exp_pw, exp_pw_mc, gamma,
        exp_proxy_logits=out.get("exp_proxy_logits"),
    )


def _pcgrad_step(model, task_losses, optimizer, scaler=None):
    per_task_grads = []
    for tl in task_losses:
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(tl).backward(retain_graph=True)
        else:
            tl.backward(retain_graph=True)
        grads = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in model.parameters()]
        per_task_grads.append(grads)
        optimizer.zero_grad(set_to_none=True)

    for i in range(len(per_task_grads)):
        for j in range(len(per_task_grads)):
            if i == j:
                continue
            dot = sum((gi * gj).sum() for gi, gj in zip(per_task_grads[i], per_task_grads[j]))
            if dot < 0:
                gj_sq = sum((gj * gj).sum() for gj in per_task_grads[j])
                if gj_sq > 1e-8:
                    coeff = (dot / gj_sq).item()
                    for k in range(len(per_task_grads[i])):
                        per_task_grads[i][k] = per_task_grads[i][k] - coeff * per_task_grads[j][k]

    for k, p in enumerate(model.parameters()):
        p.grad = sum(g[k] for g in per_task_grads) / len(per_task_grads)
    if scaler is not None:
        scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _train_epoch(
    model, dt, config, device, optimizer, window_drop, gen,
    amr_pw, exp_pw, exp_pw_mc, gamma, mixup_a,
    has_cons, cons_w, cons_t, exp_cls, accum,
    cached_weights=None, frp_pre=None, use_pcgrad=False,
    cons_interval=1,
):
    model.train()
    total_loss = 0.0
    n_train = len(dt["masks"])
    idx = _epoch_indices(n_train, dt, config, gen, cached_weights)
    bs = config.get("batch_size", 32)
    n_steps = 0
    optimizer.zero_grad(set_to_none=True)
    has_frp = frp_pre is not None
    use_amp = config.get("use_amp", False) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    div = 1.0 / accum
    non_blocking = device.type == "cuda"
    use_mixup_base = mixup_a > 0 and not has_cons and exp_cls <= 1
    is_struct = "struct_features" in dt
    is_scale = "scale_ids" in dt
    for step in range(0, n_train, bs):
        bi = idx[step: step + bs]
        B = bi.shape[0]
        mob_t = dt["mobility"][bi].to(device, non_blocking=non_blocking)
        amr_t = dt["amr"][bi].to(device, non_blocking=non_blocking)
        exp_t = dt["expansion"][bi].to(device, non_blocking=non_blocking)
        struct_b = dt["struct_features"][bi].to(device, non_blocking=non_blocking) if is_struct else None
        scale_b = dt["scale_ids"][bi].to(device, non_blocking=non_blocking) if is_scale else None
        if has_frp:
            feat = frp_pre[bi].to(device, non_blocking=non_blocking)
            raw_feat = feat
        else:
            raw_feat = dt["features"][bi].to(device, non_blocking=non_blocking)
            feat = raw_feat
        mask = dt["masks"][bi].to(device, non_blocking=non_blocking)
        feat, mask = window_drop(feat, mask, training=True)
        use_mixup = use_mixup_base and B > 1
        compute_cons = has_cons and (n_steps % cons_interval == 0)
        if compute_cons:
            cons_feat_b = dt["consistency_features"][bi].to(device=device, dtype=feat.dtype, non_blocking=non_blocking)
            cons_mask_b = dt["consistency_masks"][bi].to(device, non_blocking=non_blocking)
            cs_feat_b = dt.get("consistency_struct_features")
            cons_struct_b = (
                cs_feat_b[bi].to(device=device, dtype=feat.dtype, non_blocking=non_blocking)
                if cs_feat_b is not None else None
            )
            cs_scale = dt.get("consistency_scale_ids")
            cons_scale_b = cs_scale[bi].to(device, non_blocking=non_blocking) if cs_scale is not None else None
        with torch.amp.autocast("cuda", enabled=use_amp):
            if use_mixup:
                x, _ = model.encoder(raw_feat, mask, struct_features=struct_b, scale_ids=scale_b,
                                     frp_features=feat if has_frp else None)
                lam = torch.rand(B, device=device).mul_(2 * mixup_a).add_(1 - mixup_a).clamp_(0, 1)
                perm = torch.randperm(B, device=device)
                x_mix = lam.view(B, 1, 1) * x + (1 - lam.view(B, 1, 1)) * x[perm]
                out = model.forward_from_encoder(x_mix, mask)
                mob_oh = F.one_hot(mob_t.long(), 3).float()
                mob_oh_perm = F.one_hot(mob_t[perm].long(), 3).float()
                lm = -(lam.unsqueeze(1) * mob_oh + (1 - lam.unsqueeze(1)) * mob_oh_perm
                       ) * F.log_softmax(out["mobility_logits"], dim=-1)
                lm = lm.sum(dim=-1).mean()
                la = _focal_bce(out["amr_logits"], lam * amr_t + (1 - lam) * amr_t[perm], amr_pw, gamma)
                le = _focal_bce(out["expansion_logits"], lam * exp_t + (1 - lam) * exp_t[perm], exp_pw, gamma)
                loss = lm + la + le
            else:
                out = model(raw_feat, mask, struct_features=struct_b, scale_ids=scale_b,
                            frp_features=feat if has_frp else None)
                if compute_cons:
                    with torch.no_grad():
                        out_c = model(cons_feat_b, cons_mask_b, struct_features=cons_struct_b,
                                      scale_ids=cons_scale_b)
                    cons_loss = cons_w * _consistency_loss(out, out_c, cons_t, exp_cls)
                else:
                    cons_loss = 0.0
                losses = _compute_batch_loss(model, out, mob_t, amr_t, exp_t, amr_pw, exp_pw, exp_pw_mc, gamma)
                loss = losses["total"] + cons_loss

        scaled_loss = loss * div
        if use_mixup:
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            total_loss += loss.item()
        elif use_pcgrad and not has_cons:
            _pcgrad_step(model, [losses["mob"], losses["amr"], losses["exp"]], optimizer, scaler)
            total_loss += losses["total"].item()
        else:
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            total_loss += loss.item()

        is_last = (step + bs >= n_train)
        if is_last or (n_steps + 1) % accum == 0:
            if not (use_pcgrad and not has_cons):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        n_steps += 1
    return total_loss / max(1, n_steps)


def _run_model(data, model, device, bs=128):
    sf, sc = data.get("struct_features"), data.get("scale_ids")
    n = len(data["features"])
    dev = device if isinstance(device, torch.device) else torch.device(device)
    non_blocking = dev.type == "cuda"
    mob_logits, amr_logits, exp_logits = [], [], []
    with torch.no_grad():
        for s in range(0, n, bs):
            e = min(s + bs, n)
            o = model(data["features"][s:e].to(dev, non_blocking=non_blocking),
                      data["masks"][s:e].to(dev, non_blocking=non_blocking),
                      struct_features=sf[s:e].to(dev, non_blocking=non_blocking) if sf is not None else None,
                      scale_ids=sc[s:e].to(dev, non_blocking=non_blocking) if sc is not None else None)
            mob_logits.append(o["mobility_logits"].cpu())
            amr_logits.append(o["amr_logits"].cpu())
            exp_logits.append(o["expansion_logits"].cpu())
    return torch.cat(mob_logits).to(dev), torch.cat(amr_logits).to(dev), torch.cat(exp_logits).to(dev)


def _fit_temperature(logits, targets, loss_fn, device):
    if len(torch.unique(targets)) < 2:
        return 1.0, 0.0
    t = torch.tensor(1.0, requires_grad=True, device=device)
    b = torch.tensor(0.0, requires_grad=True, device=device)
    opt = torch.optim.LBFGS([t, b], lr=0.01, max_iter=100)
    def closure():
        opt.zero_grad()
        return loss_fn(logits * t + b, targets)
    opt.step(closure)
    return float(t.clamp(min=0.1).item()), float(b.item())


def _fit_temperature_class(logits, targets, device):
    if len(torch.unique(targets)) < 2:
        return 1.0
    t = torch.tensor(1.0, requires_grad=True, device=device)
    opt = torch.optim.LBFGS([t], lr=0.01, max_iter=100)
    def closure():
        opt.zero_grad()
        return F.cross_entropy(logits / t, targets.long())
    opt.step(closure)
    return float(t.clamp(min=0.1).item())


def fit_calibration(model, val_data, device, run_model_fn=None):
    model.eval()
    dev = device if isinstance(device, torch.device) else torch.device(device)
    mob_l, amr_l, exp_l = _run_model(val_data, model, dev) if run_model_fn is None else run_model_fn(val_data, model, dev)

    model.mob_t.data.fill_(_fit_temperature_class(mob_l, val_data["mobility"].to(dev), dev))

    at = val_data["amr"].to(dev)
    if at.dim() == 1:
        t_a, b_a = _fit_temperature(amr_l, at, F.binary_cross_entropy_with_logits, dev)
        model.amr_t.data.fill_(t_a)
        model.amr_b.data.fill_(b_a)

    if model.config.expansion_classes == 1:
        et = val_data["expansion"].to(dev)
        t_e, b_e = _fit_temperature(exp_l, et, F.binary_cross_entropy_with_logits, dev)
        model.exp_t.data.fill_(t_e)
        model.exp_b.data.fill_(b_e)

    mob_probs = torch.softmax(mob_l, dim=-1).cpu().numpy()
    amr_probs = torch.sigmoid(amr_l).cpu().numpy()
    exp_probs = (torch.softmax(exp_l, dim=-1) if model.config.expansion_classes > 1
                 else torch.sigmoid(exp_l)).cpu().numpy()
    return {
        "cal_mob_probs": mob_probs, "cal_amr_probs": amr_probs, "cal_exp_probs": exp_probs,
        "cal_mob_true": val_data["mobility"].numpy(),
        "cal_amr_true": val_data["amr"].numpy(),
        "cal_exp_true": val_data["expansion"].numpy(),
    }


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _do_train(model, train_data, val_data, config, experiment=None):
    seed = int(config.get("seed", 42))
    set_seed(seed)
    n_train = len(train_data["features"])
    n_val = len(val_data["features"])
    total = n_train + n_val
    if total > 2048:
        raise ValueError(f"Data boundary violation: train+val total {total} > 2048 (train={n_train}, val={n_val}).")
    device = torch.device(_device())
    model.to(device)
    opt = _build_optimizer(model, config)
    sched = _build_scheduler(opt, config)
    dt, dv = train_data, val_data
    artifact_dir = Path(config["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _bs, accum = config.get("batch_size", 32), config.get("gradient_accumulation_steps", 1)
    ev_int = max(1, config.get("eval_interval", 5))
    gamma = config.get("focal_loss_gamma", 2.0)
    mixup_a = config.get("mixup_alpha", 0.0)
    exp_cls = model.config.expansion_classes
    amr_pos = float(dt["amr"].sum().item() / max(1, n_train))
    amr_pw = torch.tensor([(1 - amr_pos) / max(amr_pos, 1e-6)], device=device)
    if exp_cls > 1:
        exp_c = torch.bincount(dt["expansion"].long(), minlength=exp_cls).float()
        exp_pw_mc = ((n_train - exp_c) / exp_c.clamp_min(1)).to(device)
        exp_pw_mc = (exp_pw_mc / exp_pw_mc.mean().clamp_min(1e-6)).to(device)
        exp_pw = None
    else:
        exp_pw_mc = None
        exp_pos = float(dt["expansion"].sum().item() / max(1, n_train))
        exp_pw = torch.tensor([(1 - exp_pos) / max(exp_pos, 1e-6)], device=device)
    wd = WindowDropout(config.get("dropout", 0.15))
    gen = torch.Generator().manual_seed(seed)
    cons_w = float(config.get("consistency_weight", 0.0))
    cons_t = float(config.get("consistency_temperature", 1.0))
    has_cons = cons_w > 0 and "consistency_features" in dt and exp_cls <= 1
    cached_weights = _balanced_sample_weights(dt) if config.get("balanced_sampling", False) else None
    frp_pre = None
    if (hasattr(model.encoder, "frp") and isinstance(model.encoder.frp, torch.Tensor)
            and not getattr(model.encoder, "lora_rank", 0)):
        frp_pre = (dt["features"] @ model.encoder.frp.to(dt["features"].device)).contiguous()
    use_compile = config.get("use_compile", False) and torch.cuda.is_available()
    if use_compile and hasattr(torch, "compile"):
        try:
            model.encoder = torch.compile(model.encoder, mode="reduce-overhead")
            logger.info("torch.compile enabled for encoder")
        except Exception as e:
            logger.warning("torch.compile failed: %s", e)

    if device.type == "cuda":
        pin_keys = ["features", "masks", "struct_features", "scale_ids", "mobility", "amr", "expansion"]
        if has_cons:
            pin_keys += ["consistency_features", "consistency_masks",
                         "consistency_struct_features", "consistency_scale_ids"]
        for k in pin_keys:
            if k in dt and dt[k].device.type == "cpu":
                dt[k] = dt[k].pin_memory()

    best_score, patience, history = -1.0, 0, []
    use_pcgrad = config.get("use_pcgrad", False)
    use_swa = config.get("use_swa", False)
    swa_start = max(1, config.get("swa_start_epoch", int(config.get("epochs", 120) * 0.75)))
    swa_model = None
    swa_scheduler = None
    cons_interval = max(1, config.get("consistency_interval", 1))

    for epoch in range(1, config["epochs"] + 1):
        avg_loss = _train_epoch(
            model, dt, config, device, opt, wd, gen,
            amr_pw, exp_pw, exp_pw_mc, gamma, mixup_a,
            has_cons, cons_w, cons_t, exp_cls, accum,
            cached_weights, frp_pre, use_pcgrad, cons_interval,
        )
        sched.step()
        if use_swa and epoch >= swa_start:
            if swa_model is None:
                swa_model = AveragedModel(model)
                swa_scheduler = SWALR(opt, swa_lr=config.get("lr", 1e-3) * 0.1, anneal_epochs=5)
            swa_model.update_parameters(model)
            swa_scheduler.step()
        if epoch % ev_int == 0 or epoch == 1:
            vm = evaluate(model, dv, device)
            score = _selection_score(vm, config)
            lr = opt.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch:02d}/{config['epochs']} | Loss: {avg_loss:.4f} | "
                f"AMR: {vm.get('amr_auroc', 0.0) * 100:.1f}% | "
                f"Exp: {vm.get('expansion_auroc', 0.0) * 100:.1f}% | "
                f"Mob BA: {vm.get('mobility_balanced_accuracy', 0.0) * 100:.1f}% | "
                f"Score: {score:.4f} | LR: {lr:.2e}"
            )
            history.append({"epoch": epoch, "train_loss": avg_loss, **vm})
            if experiment:
                experiment.log_metrics(epoch, train_loss=avg_loss, **vm)
            if score > best_score:
                best_score, patience = score, 0
                model.save(artifact_dir / "cassiopeia_best.pt")
            else:
                patience += 1
                if patience >= config.get("patience", 25) // ev_int:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        else:
            history.append({"epoch": epoch, "train_loss": avg_loss})
            if experiment:
                experiment.log_metrics(epoch, train_loss=avg_loss)

    model = Cassiopeia.load(artifact_dir / "cassiopeia_best.pt", device=device)
    if use_swa and swa_model is not None:
        from torch.optim.swa_utils import update_bn
        update_bn(dt, swa_model, device=device)
        swa_sd = {k.replace("module.", ""): v for k, v in swa_model.state_dict().items()
                  if k not in ("n_averaged", "module.n_averaged")}
        torch.save({"state_dict": swa_sd, "config": model.config.to_dict()},
                    artifact_dir / "cassiopeia_swa.pt")
        swa_m = Cassiopeia.load(artifact_dir / "cassiopeia_swa.pt", device=device)
        swa_metrics = evaluate(swa_m, dv, device)
        swa_score = _selection_score(swa_metrics, config)
        logger.info(f"SWA: Mob BA {swa_metrics.get('mobility_balanced_accuracy', 0.0) * 100:.2f}% | "
                     f"AMR {swa_metrics.get('amr_auroc', 0.0) * 100:.2f}% | "
                     f"Exp {swa_metrics.get('expansion_auroc', 0.0) * 100:.2f}% | "
                     f"Score {swa_score:.4f}")
        if swa_score > best_score:
            shutil.copy2(str(artifact_dir / "cassiopeia_swa.pt"),
                         str(artifact_dir / "cassiopeia_best.pt"))
    model = Cassiopeia.load(artifact_dir / "cassiopeia_best.pt", device=device)
    fit_calibration(model, dv, device)
    cal = evaluate(model, dv, device)
    logger.info(f"Calibrated: Mob BA {cal.get('mobility_balanced_accuracy', 0.0) * 100:.2f}% | "
                 f"AMR AUROC {cal.get('amr_auroc', 0.0) * 100:.2f}% | "
                 f"Exp AUROC {cal.get('expansion_auroc', 0.0) * 100:.2f}%")
    model.save(artifact_dir / "cassiopeia_best.pt")
    (artifact_dir / "cassiopeia_history.json").write_text(json.dumps(history, indent=2, default=str))
    return artifact_dir / "cassiopeia_best.pt", history


def train_cassiopeia(model, train_data, val_data, config):
    return _do_train(model, train_data, val_data, config)


def _run_fold(train_data, val_data, model_cfg, train_cfg, fold):
    cfg = {**train_cfg, "artifact_dir": str(Path(train_cfg["artifact_dir"]) / f"fold_{fold}")}
    set_seed(cfg.get("seed", 42) + fold)
    model = Cassiopeia(model_cfg)
    ckpt, _ = _do_train(model, train_data, val_data, cfg)
    model = Cassiopeia.load(ckpt, device=next(model.parameters()).device)
    return evaluate(model, val_data) | {"fold": fold}


def cross_validate(
    model_cfg: dict, data: dict, train_cfg: dict, n_folds: int = 5,
    group_ids: list[int] | None = None, save_folds: bool = False,
) -> dict:
    n = len(data["features"])
    assert n >= n_folds, f"not enough samples for {n_folds}-fold CV"
    keys = ["features", "masks", "mobility", "amr", "expansion", "struct_features", "scale_ids"]
    all_m = []
    fold_models = []
    for fold in range(n_folds):
        if group_ids is not None:
            unique = sorted(set(group_ids))
            rng = random.Random(fold)
            rng.shuffle(unique)
            fold_size = len(unique) // n_folds
            val_groups = (
                set(unique[fold * fold_size : (fold + 1) * fold_size])
                if fold < n_folds - 1
                else set(unique[fold * fold_size:])
            )
            val_idx = [i for i, g in enumerate(group_ids) if g in val_groups]
            train_idx = [i for i, g in enumerate(group_ids) if g not in val_groups]
        else:
            val_idx = list(range(fold, n, n_folds))
            train_idx = [i for i in range(n) if i not in val_idx]
        train_fold = {k: v[train_idx] for k, v in data.items() if k in keys and isinstance(v, torch.Tensor)}
        val_fold = {k: v[val_idx] for k, v in data.items() if k in keys and isinstance(v, torch.Tensor)}
        m = _run_fold(train_fold, val_fold, model_cfg, train_cfg, fold)
        all_m.append(m)
        if save_folds:
            ckpt_path = Path(train_cfg["artifact_dir"]) / f"fold_{fold}" / "cassiopeia_best.pt"
            fold_models.append(ckpt_path)
        logger.info(
            f"Fold {fold + 1}/{n_folds}: Mob BA {m.get('mobility_balanced_accuracy', 0.0) * 100:.1f}% | "
            f"AMR {m.get('amr_auroc', 0.0) * 100:.1f}% | Exp {m.get('expansion_auroc', 0.0) * 100:.1f}%"
        )
    targets = ["mobility_balanced_accuracy", "amr_auroc", "expansion_auroc", "mobility_accuracy"]
    summary = {"n_folds": n_folds}
    for k in targets:
        v = [m.get(k, 0.0) for m in all_m]
        summary[k] = {
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
        }
    scores = [_selection_score(m, {"score_mode": "equal"}) for m in all_m]
    summary["task_score"] = {"mean": float(np.mean(scores)), "std": float(np.std(scores))}
    for k, v in summary.items():
        if isinstance(v, dict) and "mean" in v:
            logger.info(f"  {k}: {v['mean'] * 100:.2f}% ± {v['std'] * 100:.2f}%")
    return summary, fold_models if save_folds else []
