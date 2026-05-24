"""
Cassiopeia training loop.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from dna_sentinel.model import _focal_bce
from dna_sentinel.utils import WindowDropout, binary_metrics, multiclass_metrics


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
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        return SequentialLR(optimizer, [
            LinearLR(optimizer, start_factor=0.1, total_iters=warmup),
            CosineAnnealingLR(optimizer, T_max=epochs - warmup, eta_min=min_lr),
        ], milestones=[warmup])
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)


def fit_calibration(model, val_data, device):
    model.eval()
    bs = 32
    n = len(val_data["features"])
    sf = val_data.get("struct_features", None)
    sc = val_data.get("scale_ids", None)
    mob_l, amr_l, exp_l = [], [], []
    with torch.no_grad():
        for s in range(0, n, bs):
            e = min(s + bs, n)
            out = model(val_data["features"][s:e].to(device),
                        val_data["masks"][s:e].to(device),
                        struct_features=sf[s:e].to(device) if sf is not None else None,
                        scale_ids=sc[s:e].to(device) if sc is not None else None)
            mob_l.append(out["mobility_logits"].cpu())
            amr_l.append(out["amr_logits"].cpu())
            exp_l.append(out["expansion_logits"].cpu())

    mob_l = torch.cat(mob_l).to(device)
    amr_l = torch.cat(amr_l).to(device)
    exp_l = torch.cat(exp_l).to(device)
    mt, at, et = (val_data[k].to(device) for k in ("mobility", "amr", "expansion"))

    t = torch.tensor(1.0, requires_grad=True, device=device)
    opt = torch.optim.LBFGS([t], lr=0.01, max_iter=50)
    def _closure():
        opt.zero_grad()
        loss = F.cross_entropy(mob_l / t, mt.long())
        loss.backward()
        return loss
    opt.step(_closure)
    model.mob_t.data.copy_(t.clamp(min=0.1))

    def _platt(logits, y):
        X = logits.cpu().numpy().reshape(-1, 1)
        yn = y.cpu().numpy()
        if len(np.unique(yn)) < 2:
            return 1.0, 0.0
        lr = LogisticRegression(C=1.0).fit(X, yn)
        return float(lr.coef_[0][0]), float(lr.intercept_[0])

    ac = model.config.amr_classes
    for i in range(max(1, ac)):
        li = amr_l if ac == 1 else amr_l[:, i]
        ti = at if ac == 1 else at[:, i]
        w, b = _platt(li, ti)
        idx = 0 if ac == 1 else i
        model.amr_w.data[idx] = w
        model.amr_b.data[idx] = b

    ec = model.config.expansion_classes
    if ec == 1:
        w, b = _platt(exp_l, et)
        model.exp_w.data[0] = w
        model.exp_b.data[0] = b


@torch.inference_mode()
def evaluate(model, data, device="cpu", batch_size=32):
    model.eval()
    n = len(data["features"])
    mob_p, amr_p, exp_p = [], [], []
    sf = data.get("struct_features", None)
    sc = data.get("scale_ids", None)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        out = model(data["features"][s:e].to(device),
                    data["masks"][s:e].to(device),
                    struct_features=sf[s:e].to(device) if sf is not None else None,
                    scale_ids=sc[s:e].to(device) if sc is not None else None)
        mob_p.append(torch.softmax(out["mobility_logits"], dim=-1).cpu())
        amr_p.append(torch.sigmoid(out["amr_logits"]).cpu())
        if model.config.expansion_classes > 1:
            exp_p.append(torch.softmax(out["expansion_logits"], dim=-1).cpu())
        else:
            exp_p.append(torch.sigmoid(out["expansion_logits"]).cpu())

    mob_np = torch.cat(mob_p).numpy()
    amr_np = torch.cat(amr_p).numpy()
    exp_np = torch.cat(exp_p).numpy()
    amr_true = data["amr"].cpu().numpy()
    exp_true = data["expansion"].cpu().numpy()

    m = multiclass_metrics(data["mobility"].cpu().numpy(), mob_np, "mobility")

    if model.config.amr_classes > 1 and amr_true.ndim > 1:
        aurocs = [binary_metrics(amr_true[:, i], amr_np[:, i], f"amr_{i}")[f"amr_{i}_auroc"]
                  for i in range(min(model.config.amr_classes, amr_true.shape[1]))
                  if len(np.unique(amr_true[:, i])) >= 2]
        m["amr_auroc"] = float(np.mean(aurocs)) if aurocs else 0.5
    else:
        m.update(binary_metrics(amr_true, amr_np, "amr"))

    if model.config.expansion_classes > 1:
        from sklearn.metrics import roc_auc_score
        aurocs = []
        n_cls = min(model.config.expansion_classes, exp_np.shape[1])
        for i in range(n_cls):
            yc = (exp_true == i).astype(float)
            if len(np.unique(yc)) >= 2 and not np.isnan(exp_np[:, i]).any():
                try:
                    aurocs.append(float(roc_auc_score(yc, exp_np[:, i])))
                except ValueError:
                    pass
        m["expansion_auroc"] = float(np.mean(aurocs)) if aurocs else 0.5
        m["expansion_accuracy"] = float(np.mean(exp_true == exp_np.argmax(axis=1)))
    else:
        m.update(binary_metrics(exp_true, exp_np, "expansion"))

    return m


def _aux_loss(model, out, mob_t, amr_t):
    if not model.training or model.config.aux_loss_weight <= 0:
        return 0.0
    loss = 0.0
    if "aux_mob_logits" in out:
        loss = loss + F.cross_entropy(out["aux_mob_logits"], mob_t.long())
    if "aux_amr_logits" in out:
        loss = loss + F.binary_cross_entropy_with_logits(out["aux_amr_logits"], amr_t)
    return model.config.aux_loss_weight * loss


def _uncertainty_weighted(lm, la, le, log_vars):
    lv = log_vars.clamp(-6, 6)
    losses = torch.stack([lm, la, le])
    return (0.5 * torch.exp(-lv) * losses + 0.5 * lv).sum()


def train_cassiopeia(model, train_data, val_data, config):
    torch.manual_seed(config.get("seed", 42))
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config)

    n_train = len(train_data["features"])
    dt, dv = train_data, val_data
    artifact_dir = Path(config["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    bs, accum = config.get("batch_size", 32), config.get("gradient_accumulation_steps", 1)
    gamma = config.get("focal_loss_gamma", 2.0)
    mixup_a = config.get("mixup_alpha", 0.0)
    exp_cls = model.config.expansion_classes

    amr_pos = float(dt["amr"].sum().item() / max(1, n_train))
    amr_pw = torch.tensor([(1 - amr_pos) / max(amr_pos, 1e-6)], device=device)

    if exp_cls > 1:
        exp_counts = torch.bincount(dt["expansion"].long(), minlength=exp_cls).float()
        raw_pw = (n_train - exp_counts) / exp_counts.clamp_min(1)
        exp_pw_mc = (raw_pw / raw_pw.mean().clamp_min(1e-6)).to(device)
    else:
        exp_pw_mc = None
        exp_pos = float(dt["expansion"].sum().item() / max(1, n_train))
        exp_pw = torch.tensor([(1 - exp_pos) / max(exp_pos, 1e-6)], device=device)

    best_score, patience, history = -1.0, 0, []
    optimizer.zero_grad(set_to_none=True)
    window_drop = WindowDropout(config.get("dropout", 0.15))

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss, n_steps = 0.0, 0
        idx = torch.randperm(n_train)
        for step in range(0, n_train, bs):
            bi = idx[step: step + bs].tolist()
            feat = dt["features"][bi].to(device)
            mask = dt["masks"][bi].to(device)
            feat, mask = window_drop(feat, mask, training=True)
            mob_t = dt["mobility"][bi].to(device)
            amr_t = dt["amr"][bi].to(device)
            exp_t = dt["expansion"][bi].to(device)
            struct_b = dt["struct_features"][bi].to(device) if "struct_features" in dt else None
            scale_b = dt["scale_ids"][bi].to(device) if "scale_ids" in dt else None

            B = len(bi)
            if mixup_a > 0 and B > 1 and exp_cls <= 1:
                x, aux_features = model.encoder(feat, mask, struct_features=struct_b, scale_ids=scale_b)
                lam = torch.rand(B, device=device).mul_(2 * mixup_a).add_(1 - mixup_a).clamp_(0, 1)
                perm = torch.randperm(B, device=device)
                lam_v = lam.view(B, 1, 1)
                x_mix = lam_v * x + (1 - lam_v) * x[perm]
                mix_mask = mask & mask[perm]
                out = model.forward_from_encoder(x_mix, mix_mask, aux_features)
                lm = -(lam.unsqueeze(1) * F.one_hot(mob_t.long(), 3).float()
                       + (1 - lam.unsqueeze(1)) * F.one_hot(mob_t[perm].long(), 3).float()
                       ) * F.log_softmax(out["mobility_logits"], dim=-1)
                lm = lm.sum(dim=-1).mean()
                amr_mix = lam * amr_t.float() + (1 - lam) * amr_t[perm].float()
                la = _focal_bce(out["amr_logits"], amr_mix, amr_pw, gamma)
                exp_mix = lam * exp_t.float() + (1 - lam) * exp_t[perm].float()
                le = _focal_bce(out["expansion_logits"].squeeze(-1), exp_mix, exp_pw, gamma)
            else:
                out = model(feat, mask, struct_features=struct_b, scale_ids=scale_b)
                lm = F.cross_entropy(out["mobility_logits"], mob_t.long(), label_smoothing=model.config.label_smoothing)
                pw = amr_pw if exp_cls <= 1 else None
                la = _focal_bce(out["amr_logits"], amr_t, pw, gamma)
                if exp_cls > 1:
                    le = F.cross_entropy(out["expansion_logits"], exp_t.long(), weight=exp_pw_mc,
                                          label_smoothing=model.config.label_smoothing)
                else:
                    le = _focal_bce(out["expansion_logits"].squeeze(-1), exp_t, exp_pw, gamma)

            aux_loss_val = _aux_loss(model, out, mob_t, amr_t)
            loss = _uncertainty_weighted(lm, la, le, model.log_vars) + aux_loss_val
            loss = loss / accum

            loss.backward()
            total_loss += loss.item() * accum
            n_steps += 1

            if (step // bs + 1) % accum == 0 or step + bs >= n_train:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        avg_loss = total_loss / max(1, n_steps)
        val_metrics = evaluate(model, dv, device)

        score = (val_metrics.get("amr_auroc", 0.0) + val_metrics.get("expansion_auroc", 0.0)
                 + 2.0 * val_metrics.get("mobility_balanced_accuracy", 0.0))
        print(f"Epoch {epoch:02d}/{config['epochs']:02d} | Loss: {avg_loss:.4f} | "
              f"AMR: {val_metrics.get('amr_auroc', 0.0)*100:.1f}% | "
              f"Exp: {val_metrics.get('expansion_auroc', 0.0)*100:.1f}% | "
              f"Mob BA: {val_metrics.get('mobility_balanced_accuracy', 0.0)*100:.1f}% | Score: {score:.4f}")
        history.append({"epoch": epoch, "train_loss": avg_loss, **val_metrics})

        if score > best_score:
            best_score, patience = score, 0
            model.save(artifact_dir / "cassiopeia_best.pt")
        else:
            patience += 1
            if patience >= config.get("patience", 25):
                print(f"Early stopping at epoch {epoch}")
                break

    model_class = type(model)
    model = model_class.load(artifact_dir / "cassiopeia_best.pt", device=device)
    fit_calibration(model, dv, device)
    calibrated = evaluate(model, dv, device)
    print(f"\nCalibrated: Mob BA {calibrated.get('mobility_balanced_accuracy', 0.0)*100:.2f}% | "
          f"AMR AUROC {calibrated.get('amr_auroc', 0.0)*100:.2f}% | "
          f"Exp AUROC {calibrated.get('expansion_auroc', 0.0)*100:.2f}%")
    model.save(artifact_dir / "cassiopeia_best.pt")
    (artifact_dir / "cassiopeia_history.json").write_text(json.dumps(history, indent=2, default=str))
    return artifact_dir / "cassiopeia_best.pt", history
