from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from dna_sentinel.model import Cassiopeia, _focal_bce
from dna_sentinel.utils import WindowDropout, binary_metrics, logger, multiclass_metrics, set_seed


def _inverse_label_frequency(labels: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    y = labels.detach().cpu().long().view(-1)
    mx = int(y.max().item()) + 1 if y.numel() else 1
    counts = torch.bincount(y, minlength=mx).float()
    total = counts.sum()
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
    return (mob + amr + exp) / 3.0


def _build_optimizer(model, config):
    backbone_lr = config.get("backbone_lr", config.get("lr", 1e-3))
    head_lr = config.get("head_lr", config.get("lr", 1e-3))
    backbone, heads = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (heads if n.startswith(("mob_", "amr_", "exp_", "log_vars")) else backbone).append(p)
    groups = [{"params": g, "lr": lr} for g, lr in [(backbone, backbone_lr), (heads, head_lr)] if g]
    return torch.optim.AdamW(groups, lr=config["lr"], weight_decay=config.get("weight_decay", 0.05))


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


def _consistency_loss(out_ref: dict, out_aug: dict, temperature: float = 1.0) -> torch.Tensor:
    t = max(float(temperature), 1e-6)
    mob_ref = torch.softmax(out_ref["mobility_logits"].detach() / t, dim=-1)
    mob = F.kl_div(torch.log_softmax(out_aug["mobility_logits"] / t, dim=-1), mob_ref, reduction="batchmean") * (t * t)
    amr_ref = torch.sigmoid(out_ref["amr_logits"].detach())
    exp_ref = torch.sigmoid(out_ref["expansion_logits"].detach())
    return (
        mob
        + F.mse_loss(torch.sigmoid(out_aug["amr_logits"]), amr_ref)
        + F.mse_loss(torch.sigmoid(out_aug["expansion_logits"]), exp_ref)
    )


@torch.inference_mode()
def evaluate(model, data, device="cpu", batch_size=32, return_probs=False):
    model.eval()
    n = len(data["features"])
    mob_p, amr_p, exp_p = [], [], []
    sf = data.get("struct_features", None)
    sc = data.get("scale_ids", None)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        out = model(
            data["features"][s:e].to(device),
            data["masks"][s:e].to(device),
            struct_features=sf[s:e].to(device) if sf is not None else None,
            scale_ids=sc[s:e].to(device) if sc is not None else None,
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
        aux_mob_logits=out.get("aux_mob_logits"),
        aux_amr_logits=out.get("aux_amr_logits"),
        exp_proxy_logits=out.get("exp_proxy_logits"),
    )


def _pcgrad_step(model, task_losses, optimizer, accum, scaler=None):
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
):
    model.train()
    total_loss = 0.0
    n_train = len(dt["features"])
    idx = _epoch_indices(n_train, dt, config, gen, cached_weights)
    bs = config.get("batch_size", 32)
    n_steps = 0
    optimizer.zero_grad(set_to_none=True)
    has_frp = frp_pre is not None
    use_amp = config.get("use_amp", False) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_device = "cuda" if use_amp else "cpu"
    for step in range(0, n_train, bs):
        bi = idx[step : step + bs].tolist()
        raw_feat = dt["features"][bi].to(device)
        mask = dt["masks"][bi].to(device)
        feat = (frp_pre[bi].to(device) if has_frp else raw_feat)
        feat, mask = window_drop(feat, mask, training=True)
        mob_t = dt["mobility"][bi].to(device)
        amr_t = dt["amr"][bi].to(device)
        exp_t = dt["expansion"][bi].to(device)
        B = len(bi)
        struct_b = dt["struct_features"][bi].to(device) if "struct_features" in dt else None
        scale_b = dt["scale_ids"][bi].to(device) if "scale_ids" in dt else None
        use_mixup = mixup_a > 0 and not has_cons and B > 1 and exp_cls <= 1
        cons_feat_b = dt["consistency_features"][bi].to(device=device, dtype=feat.dtype) if has_cons else None
        cons_mask_b = dt["consistency_masks"][bi].to(device) if has_cons else None
        cs_feat_b = dt.get("consistency_struct_features")
        cons_struct_b = cs_feat_b[bi].to(device=device, dtype=feat.dtype) if has_cons and cs_feat_b is not None else None
        cs_scale = dt.get("consistency_scale_ids")
        cons_scale_b = cs_scale[bi].to(device) if has_cons and cs_scale is not None else None
        with torch.amp.autocast(amp_device, enabled=use_amp):
            if use_mixup:
                enc_kw = dict(struct_features=struct_b, scale_ids=scale_b, frp_features=feat if has_frp else None)
                x, aux_f = model.encoder(raw_feat, mask, **enc_kw)
                lam = torch.rand(B, device=device).mul_(2 * mixup_a).add_(1 - mixup_a).clamp_(0, 1)
                perm = torch.randperm(B, device=device)
                x_mix = lam.view(B, 1, 1) * x + (1 - lam.view(B, 1, 1)) * x[perm]
                out = model.forward_from_encoder(x_mix, mask, aux_f)
                l_mob = -(
                    lam.unsqueeze(1) * F.one_hot(mob_t.long(), 3).float()
                    + (1 - lam.unsqueeze(1)) * F.one_hot(mob_t[perm].long(), 3).float()
                ) * F.log_softmax(out["mobility_logits"], dim=-1)
                lm = l_mob.sum(dim=-1).mean()
                la = _focal_bce(out["amr_logits"], lam * amr_t.float() + (1 - lam) * amr_t[perm].float(), amr_pw, gamma)
                le = _focal_bce(
                    out["expansion_logits"], lam * exp_t.float() + (1 - lam) * exp_t[perm].float(), exp_pw, gamma
                )
                loss = lm + la + le
            else:
                out = model(raw_feat, mask, struct_features=struct_b, scale_ids=scale_b, frp_features=feat if has_frp else None)
                if has_cons:
                    with torch.inference_mode():
                        out_c = model(cons_feat_b, cons_mask_b, struct_features=cons_struct_b, scale_ids=cons_scale_b)
                    cons_loss = cons_w * _consistency_loss(out, out_c, cons_t)
                else:
                    cons_loss = 0.0
                losses = _compute_batch_loss(model, out, mob_t, amr_t, exp_t, amr_pw, exp_pw, exp_pw_mc, gamma)
                loss = losses["total"] + cons_loss

        if use_mixup:
            scaler.scale(loss / accum).backward()
            total_loss += loss.item()
        elif use_pcgrad and model.training and cons_loss == 0:
            _pcgrad_step(model, [losses["mob"], losses["amr"], losses["exp"]], optimizer, accum, scaler)
            total_loss += losses["total"].item()
        else:
            scaler.scale(loss / accum).backward()
            total_loss += loss.item()

        if step // bs + 1 >= n_train // bs or (step // bs + 1) % accum == 0:
            if not (use_pcgrad and model.training and cons_loss == 0):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        n_steps += 1
    return total_loss / max(1, n_steps)


def _run_model(data, model, device, bs=32):
    sf, sc = data.get("struct_features"), data.get("scale_ids")
    n = len(data["features"])
    mob_logits, amr_logits, exp_logits = [], [], []
    with torch.no_grad():
        for s in range(0, n, bs):
            i = min(s + bs, n)
            o = model(data["features"][s:i].to(device), data["masks"][s:i].to(device),
                      struct_features=sf[s:i].to(device) if sf is not None else None,
                      scale_ids=sc[s:i].to(device) if sc is not None else None)
            mob_logits.append(o["mobility_logits"].cpu())
            amr_logits.append(o["amr_logits"].cpu())
            exp_logits.append(o["expansion_logits"].cpu())
    return (torch.cat(mob_logits).to(device), torch.cat(amr_logits).to(device), torch.cat(exp_logits).to(device))


def fit_calibration(model, val_data, device, nonplasmid_data=None, data_dir=None):
    model.eval()
    model.amr_w.data.fill_(1.0)
    model.amr_b.data.fill_(0.0)
    model.exp_w.data.fill_(1.0)
    model.exp_b.data.fill_(0.0)
    model.exp_t.data.fill_(1.0)
    model.mob_t.data.fill_(1.0)
    if nonplasmid_data is None and data_dir is not None:
        np_path = Path(data_dir) / "nonplasmid_control_features.pt"
        if np_path.exists():
            try:
                from dna_sentinel.cli import _load_data
                nonplasmid_data = _load_data(Path(data_dir), "nonplasmid_control", model.config.n_structural_features)
                logger.info(f"Loaded {len(nonplasmid_data['features'])} non-plasmid samples for expansion calibration")
            except Exception:
                pass
    mob_l, amr_l, exp_l = _run_model(val_data, model, device)
    mt, at, et = (val_data[k].to(device) for k in ("mobility", "amr", "expansion"))
    if len(torch.unique(mt)) >= 2:
        t = torch.tensor(1.0, requires_grad=True, device=device)
        opt = torch.optim.LBFGS([t], lr=0.01, max_iter=50)

        def _closure():
            opt.zero_grad()
            return F.cross_entropy(mob_l / t, mt.long())

        opt.step(_closure)
        model.mob_t.data.copy_(t.clamp(min=0.1))
    for i in range(max(1, model.config.amr_classes)):
        li = amr_l if model.config.amr_classes == 1 else amr_l[:, i]
        ti = at if model.config.amr_classes == 1 else at[:, i]
        X, yn = li.cpu().numpy().reshape(-1, 1), ti.cpu().numpy()
        if len(np.unique(yn)) >= 2:
            lr = LogisticRegression(C=1.0).fit(X, yn)
            idx = 0 if model.config.amr_classes == 1 else i
            model.amr_w.data[idx] = float(lr.coef_[0][0])
            model.amr_b.data[idx] = float(lr.intercept_[0])
    if model.config.expansion_classes == 1:
        _, _, exp_l = _run_model(val_data, model, device)
        X_e, yn_e = exp_l.cpu().numpy().reshape(-1, 1), et.cpu().numpy()
        if len(np.unique(yn_e)) >= 2:
            lr_e = LogisticRegression(C=1.0).fit(X_e, yn_e)
            model.exp_w.data.fill_(float(lr_e.coef_[0][0]))
            model.exp_b.data.fill_(float(lr_e.intercept_[0]))
        model.exp_t.data.fill_(1.0)


def _apply_swa(model, swa_checkpoints):
    if len(swa_checkpoints) < 2:
        return
    swa_state = deepcopy(swa_checkpoints[0])
    for key in swa_state:
        for i in range(1, len(swa_checkpoints)):
            swa_state[key] = swa_state[key] + swa_checkpoints[i][key]
        swa_state[key] = swa_state[key] / len(swa_checkpoints)
    model.load_state_dict(swa_state)
    logger.info(f"SWA applied over {len(swa_checkpoints)} checkpoints")


def _do_train(model, train_data, val_data, config, experiment=None):
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    opt = _build_optimizer(model, config)
    sched = _build_scheduler(opt, config)
    n_train = len(train_data["features"])
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
    if hasattr(model.encoder, "frp") and isinstance(model.encoder.frp, torch.Tensor):
        frp_pre = (dt["features"] @ model.encoder.frp.to(dt["features"].device)).contiguous()
    best_score, patience, history = -1.0, 0, []
    use_pcgrad = config.get("use_pcgrad", False)
    use_swa = config.get("use_swa", False)
    swa_start = max(1, config.get("swa_start_epoch", int(config.get("epochs", 120) * 0.75)))
    swa_checkpoints = []

    for epoch in range(1, config["epochs"] + 1):
        avg_loss = _train_epoch(
            model, dt, config, device, opt, wd, gen,
            amr_pw, exp_pw, exp_pw_mc, gamma, mixup_a,
            has_cons, cons_w, cons_t, exp_cls, accum,
            cached_weights, frp_pre, use_pcgrad,
        )
        sched.step()
        if epoch % ev_int == 0 or epoch == 1:
            vm = evaluate(model, dv, device)
            score = _selection_score(vm, config)
            lr = opt.param_groups[0]['lr']
            mob = vm.get("mobility_balanced_accuracy", 0.0) * 100
            amr = vm.get("amr_auroc", 0.0) * 100
            exp = vm.get("expansion_auroc", 0.0) * 100
            logger.info(
                f"Epoch {epoch:02d}/{config['epochs']} | Loss: {avg_loss:.4f} | "
                f"AMR: {amr:.1f}% | Exp: {exp:.1f}% | Mob BA: {mob:.1f}% | "
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

        if use_swa and epoch >= swa_start:
            swa_checkpoints.append(deepcopy(model.state_dict()))

    model = Cassiopeia.load(artifact_dir / "cassiopeia_best.pt", device=device)

    if use_swa and swa_checkpoints:
        _apply_swa(model, swa_checkpoints)
        model.save(artifact_dir / "cassiopeia_swa.pt")
        swa_metrics = evaluate(model, dv, device)
        swa_score = _selection_score(swa_metrics, config)
        logger.info(f"SWA: Mob BA {swa_metrics.get('mobility_balanced_accuracy', 0.0) * 100:.2f}% | "
                     f"AMR {swa_metrics.get('amr_auroc', 0.0) * 100:.2f}% | "
                     f"Exp {swa_metrics.get('expansion_auroc', 0.0) * 100:.2f}% | "
                     f"Score {swa_score:.4f}")
        if swa_score > best_score:
            model.save(artifact_dir / "cassiopeia_best.pt")
            logger.info("SWA checkpoint promoted to best")

    model = Cassiopeia.load(artifact_dir / "cassiopeia_best.pt", device=device)
    data_dir = config.get("data_dir", artifact_dir.parent.parent / "data" / "dna_sentinel")
    fit_calibration(model, dv, device, data_dir=data_dir)
    cal = evaluate(model, dv, device)
    mob = cal.get("mobility_balanced_accuracy", 0.0) * 100
    amr = cal.get("amr_auroc", 0.0) * 100
    exp = cal.get("expansion_auroc", 0.0) * 100
    logger.info(f"Calibrated: Mob BA {mob:.2f}% | AMR AUROC {amr:.2f}% | Exp AUROC {exp:.2f}%")
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
    model = Cassiopeia.load(ckpt)
    return evaluate(model, val_data) | {"fold": fold}


def cross_validate(model_cfg: dict, data: dict, train_cfg: dict, n_folds: int = 5, group_ids: list[int] | None = None, save_folds: bool = False) -> dict:
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
            val_groups = set(unique[fold * fold_size : (fold + 1) * fold_size]) if fold < n_folds - 1 else set(unique[fold * fold_size:])
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
