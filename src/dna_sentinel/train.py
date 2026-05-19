import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from dna_sentinel.utils import WindowDropout, binary_metrics, multiclass_metrics


def fit_temperature(model, val_data, device):
    model.eval()
    feats, specs, masks, sids = val_data["features"], val_data["spec_features"], val_data["masks"], val_data["scale_ids"]
    n, batch_size = len(feats), 32
    mob_list, amr_list, exp_list = [], [], []
    with torch.no_grad():
        model.amr_calib_w.fill_(1.0)
        model.amr_calib_b.fill_(0.0)
        model.exp_calib_w.fill_(1.0)
        model.exp_calib_b.fill_(0.0)
        model.mobility_calib_t.fill_(1.0)
        for s in range(0, n, batch_size):
            e = s + batch_size
            out = model(feats[s:e].to(device), specs[s:e].to(device), masks[s:e].to(device), sids[s:e].to(device))
            mob_list.append(out["mobility_logits"].cpu())
            amr_list.append(out["amr_logits"].cpu())
            exp_list.append(out["expansion_logits"].cpu())

    mob_logits, amr_logits, exp_logits = torch.cat(mob_list), torch.cat(amr_list), torch.cat(exp_list)
    mob_targets, amr_targets, exp_targets = val_data["mobility"].cpu(), val_data["amr"].cpu(), val_data["expansion"].cpu()

    def optimize_t(logits, targets):
        t = torch.ones(1, requires_grad=True)
        opt = torch.optim.LBFGS([t], lr=0.01, max_iter=50)
        def eval_loss():
            opt.zero_grad()
            loss = F.cross_entropy(logits / t, targets.long())
            loss.backward()
            return loss
        opt.step(eval_loss)
        return max(0.1, float(t.item()))

    t_mob = optimize_t(mob_logits, mob_targets)

    def optimize_platt(logits, targets):
        X, y = logits.numpy().reshape(-1, 1), targets.numpy()
        if len(np.unique(y)) < 2:
            return 1.0, 0.0
        lr = LogisticRegression(C=1e5).fit(X, y)
        return float(lr.coef_[0][0]), float(lr.intercept_[0])

    w_amr, b_amr = optimize_platt(amr_logits, amr_targets)
    w_exp, b_exp = optimize_platt(exp_logits, exp_targets)
    model.amr_calib_w.copy_(torch.tensor(w_amr, device=device))
    model.amr_calib_b.copy_(torch.tensor(b_amr, device=device))
    model.exp_calib_w.copy_(torch.tensor(w_exp, device=device))
    model.exp_calib_b.copy_(torch.tensor(b_exp, device=device))
    model.mobility_calib_t.copy_(torch.tensor(t_mob, device=device))


def binary_focal_loss(logits, targets, pos_weight=None, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    loss = ((1.0 - p_t) ** gamma) * bce
    return loss.mean()


def update_ema(student, teacher, beta):
    with torch.no_grad():
        for s_param, t_param in zip(student.parameters(), teacher.parameters()):
            t_param.data.mul_(beta).add_(s_param.data, alpha=1.0 - beta)


def train_kmer_transformer(model, train_data, val_data, config):
    torch.manual_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    ema_model = copy.deepcopy(model).to(device).eval()
    for p in ema_model.parameters():
        p.requires_grad = False

    ema_beta = config.get("ema_beta", 0.999)
    backbone_lr = config.get("backbone_lr", config["lr"])
    head_lr = config.get("head_lr", config["lr"])

    def is_head(name):
        return any(h in name for h in ["head", "logit_scale", "log_vars"])

    def is_no_decay(name):
        return any(nd in name for nd in ["bias", "norm", "scale", "log_vars"])

    param_groups = []

    backbone_decay = [p for n, p in model.named_parameters() if p.requires_grad and not is_head(n) and not is_no_decay(n)]
    if backbone_decay:
        param_groups.append({"params": backbone_decay, "lr": backbone_lr, "weight_decay": config["weight_decay"]})

    backbone_no_decay = [p for n, p in model.named_parameters() if p.requires_grad and not is_head(n) and is_no_decay(n)]
    if backbone_no_decay:
        param_groups.append({"params": backbone_no_decay, "lr": backbone_lr, "weight_decay": 0.0})

    heads_decay = [p for n, p in model.named_parameters() if p.requires_grad and is_head(n) and not is_no_decay(n)]
    if heads_decay:
        param_groups.append({"params": heads_decay, "lr": head_lr, "weight_decay": config["weight_decay"]})

    heads_no_decay = [p for n, p in model.named_parameters() if p.requires_grad and is_head(n) and is_no_decay(n)]
    if heads_no_decay:
        param_groups.append({"params": heads_no_decay, "lr": head_lr, "weight_decay": 0.0})

    optimizer = torch.optim.AdamW(param_groups)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-5)
    window_dropout = WindowDropout(config.get("window_dropout", 0.25))
    n_train = len(train_data["features"])
    device_train = {k: v.to(device) for k, v in train_data.items()}
    device_val = {k: v.to(device) for k, v in val_data.items()}
    artifact_dir = Path(config["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    mob_counts = torch.bincount(device_train["mobility"].long(), minlength=3).float().clamp(min=1)
    mob_weight = mob_counts.sum() / (3 * mob_counts)
    amr_pos = float(device_train["amr"].sum().item() / max(1, n_train))
    exp_pos = float(device_train["expansion"].sum().item() / max(1, n_train))
    amr_pos_weight = torch.tensor([(1 - amr_pos) / max(amr_pos, 1e-6)], device=device)
    exp_pos_weight = torch.tensor([(1 - exp_pos) / max(exp_pos, 1e-6)], device=device)

    best_score, patience_counter, history = -1.0, 0, []
    loss_buffer = {"mob": [], "amr": [], "exp": []}
    w_mob, w_amr, w_exp = 1.0, 1.0, 1.0

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        losses = []
        losses_mob_epoch = []
        losses_amr_epoch = []
        losses_exp_epoch = []
        idx = torch.randperm(n_train, device=device)
        for start in range(0, n_train, config["batch_size"]):
            bi = idx[start:start + config["batch_size"]]
            feat, spec, mask, sid = device_train["features"][bi], device_train["spec_features"][bi], device_train["masks"][bi], device_train["scale_ids"][bi]
            mob, amr, exp = device_train["mobility"][bi], device_train["amr"][bi], device_train["expansion"][bi]

            [feat1, spec1], mask1 = window_dropout([feat, spec], mask, training=True)
            out1 = model(feat1, spec1, mask1, sid)
            [feat2, spec2], mask2 = window_dropout([feat, spec], mask, training=True)
            out2 = model(feat2, spec2, mask2, sid)

            loss_mob = F.cross_entropy(out1["mobility_logits"], mob.long(), weight=mob_weight, label_smoothing=0.1)
            loss_amr = binary_focal_loss(out1["amr_logits"], amr, pos_weight=amr_pos_weight, gamma=2.0)
            loss_exp = binary_focal_loss(out1["expansion_logits"], exp, pos_weight=exp_pos_weight, gamma=2.0)

            loss_task = w_mob * loss_mob + w_amr * loss_amr + w_exp * loss_exp

            losses_mob_epoch.append(loss_mob.item())
            losses_amr_epoch.append(loss_amr.item())
            losses_exp_epoch.append(loss_exp.item())

            with torch.no_grad():
                ema_out = ema_model(feat1, spec1, mask1, sid)

            targets_amr_distill = 0.9 * torch.sigmoid(ema_out["amr_logits"]) + 0.05
            targets_exp_distill = 0.9 * torch.sigmoid(ema_out["expansion_logits"]) + 0.05
            targets_mob_distill = 0.9 * F.softmax(ema_out["mobility_logits"], dim=-1) + 0.1 / 3.0

            loss_distill_mob = F.kl_div(F.log_softmax(out1["mobility_logits"], dim=-1), targets_mob_distill, reduction="batchmean")
            loss_distill_amr = F.binary_cross_entropy_with_logits(out1["amr_logits"], targets_amr_distill)
            loss_distill_exp = F.binary_cross_entropy_with_logits(out1["expansion_logits"], targets_exp_distill)

            loss_cl = (1.0 - F.cosine_similarity(out1["pooled"], out2["pooled"], dim=-1)).mean()
            entropy = -(out1["evidence_weights"].clamp_min(1e-8) * out1["evidence_weights"].clamp_min(1e-8).log()).sum(dim=1).mean()

            loss = loss_task + 0.5 * (loss_distill_mob + loss_distill_amr + loss_distill_exp) + 0.1 * loss_cl + 0.005 * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update_ema(model, ema_model, ema_beta)
            losses.append(loss.item())

        avg_mob_loss = sum(losses_mob_epoch) / len(losses_mob_epoch)
        avg_amr_loss = sum(losses_amr_epoch) / len(losses_amr_epoch)
        avg_exp_loss = sum(losses_exp_epoch) / len(losses_exp_epoch)

        loss_buffer["mob"].append(avg_mob_loss)
        loss_buffer["amr"].append(avg_amr_loss)
        loss_buffer["exp"].append(avg_exp_loss)

        if len(loss_buffer["mob"]) >= 2:
            r_mob = loss_buffer["mob"][-1] / (loss_buffer["mob"][-2] + 1e-8)
            r_amr = loss_buffer["amr"][-1] / (loss_buffer["amr"][-2] + 1e-8)
            r_exp = loss_buffer["exp"][-1] / (loss_buffer["exp"][-2] + 1e-8)

            r_tensor = torch.tensor([r_mob, r_amr, r_exp])
            weights = 3.0 * F.softmax(r_tensor / 2.0, dim=0)
            w_mob, w_amr, w_exp = weights[0].item(), weights[1].item(), weights[2].item()

        scheduler.step()
        avg_loss = sum(losses) / len(losses)
        val_metrics = evaluate_kmer_transformer(ema_model, device_val, device)
        row = {"epoch": epoch, "train_loss": avg_loss, **val_metrics}
        history.append(row)
        score = val_metrics.get("amr_auroc", 0.0) + val_metrics.get("expansion_auroc", 0.0) + 2.0 * val_metrics.get("mobility_balanced_accuracy", 0.0)

        print(
            f"Epoch {epoch:02d}/{config['epochs']:02d} | "
            f"Loss: {avg_loss:.4f} | "
            f"AMR AUROC: {val_metrics.get('amr_auroc', 0.0):.4f} | "
            f"Exp AUROC: {val_metrics.get('expansion_auroc', 0.0):.4f} | "
            f"Mob Acc: {val_metrics.get('mobility_balanced_accuracy', 0.0):.4f} | "
            f"Score: {score:.4f}"
        )

        if score > best_score:
            best_score, patience_counter = score, 0
            ema_model.save(artifact_dir / "kmer_transformer_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= config.get("patience", 25):
                print(f"Early stopping at epoch {epoch}")
                break

    model = model.load(artifact_dir / "kmer_transformer_best.pt", device=device)
    fit_temperature(model, device_val, device)
    model.save(artifact_dir / "kmer_transformer_best.pt")
    (artifact_dir / "kmer_transformer_history.json").write_text(json.dumps(history, indent=2))
    return artifact_dir / "kmer_transformer_best.pt", history


@torch.inference_mode()
def evaluate_kmer_transformer(model, data, device="cpu", batch_size: int = 32):
    model.eval()
    model.to(device)
    feats, specs, masks, sids = data["features"], data["spec_features"], data["masks"], data["scale_ids"]
    n = len(feats)
    mob_list, amr_list, exp_list = [], [], []
    for start in range(0, n, batch_size):
        end = start + batch_size
        out = model(feats[start:end].to(device), specs[start:end].to(device), masks[start:end].to(device), sids[start:end].to(device))
        mob_list.append(torch.softmax(out["mobility_logits"], dim=-1).cpu())
        amr_list.append(torch.sigmoid(out["amr_logits"]).cpu())
        exp_list.append(torch.sigmoid(out["expansion_logits"]).cpu())
    mob_p = torch.cat(mob_list).numpy() if mob_list else np.zeros((0, 3))
    amr_p = torch.cat(amr_list).numpy() if amr_list else np.zeros((0,))
    exp_p = torch.cat(exp_list).numpy() if exp_list else np.zeros((0,))
    m = {}
    m.update(multiclass_metrics(data["mobility"].cpu().numpy(), mob_p, "mobility"))
    m.update(binary_metrics(data["amr"].cpu().numpy(), amr_p, "amr"))
    m.update(binary_metrics(data["expansion"].cpu().numpy(), exp_p, "expansion"))
    return m
