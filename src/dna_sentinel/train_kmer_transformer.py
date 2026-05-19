"""KmerTransformer training and evaluation loop with PCGrad gradient surgery."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from dna_sentinel.augmentation import WindowDropout
from dna_sentinel.metrics import binary_metrics, multiclass_metrics


def train_kmer_transformer(model, train_data, val_data, config):
    torch.manual_seed(config.get("seed", 42))
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "scale" in name or "bias" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": config["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0}
    ], lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-5)
    window_dropout = WindowDropout(config.get("window_dropout", 0.25))
    artifact_dir = Path(config["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    device_train = {k: v.to(device) for k, v in train_data.items()}
    device_val = {k: v.to(device) for k, v in val_data.items()}

    mob_counts = torch.bincount(device_train["mobility"].long(), minlength=3).float().clamp(min=1)
    mob_weight = (mob_counts.sum() / (3 * mob_counts))
    n_train = len(device_train["features"])
    amr_pos = float(device_train["amr"].sum().item() / max(1, n_train))
    exp_pos = float(device_train["expansion"].sum().item() / max(1, n_train))
    amr_pos_weight = torch.tensor([(1 - amr_pos) / max(amr_pos, 1e-6)], device=device)
    exp_pos_weight = torch.tensor([(1 - exp_pos) / max(exp_pos, 1e-6)], device=device)
    best_score = -1.0
    patience_counter = 0
    history = []
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        losses = []
        idx = torch.randperm(n_train, device=device)
        for start in range(0, n_train, config["batch_size"]):
            bi = idx[start:start + config["batch_size"]]
            feat = device_train["features"][bi]
            spec = device_train["spec_features"][bi]
            mask = device_train["masks"][bi]
            sid = device_train["scale_ids"][bi]
            mob = device_train["mobility"][bi]
            amr = device_train["amr"][bi]
            exp = device_train["expansion"][bi]

            [feat1, spec1], mask1 = window_dropout([feat, spec], mask, training=True)
            out1 = model(feat1, spec1, mask1, sid)
            [feat2, spec2], mask2 = window_dropout([feat, spec], mask, training=True)
            out2 = model(feat2, spec2, mask2, sid)

            p1 = F.normalize(out1["pooled"], p=2, dim=-1)
            p2 = F.normalize(out2["pooled"], p=2, dim=-1)
            sim = torch.mm(p1, p2.t()) / 0.1
            labels = torch.arange(p1.shape[0], device=device)
            loss_cl = F.cross_entropy(sim, labels)
            entropy = -(out1["evidence_weights"].clamp_min(1e-8) * out1["evidence_weights"].clamp_min(1e-8).log()).sum(dim=1).mean()
            aux = 0.1 * loss_cl + 0.005 * entropy

            loss_mob = F.cross_entropy(out1["mobility_logits"], mob, weight=mob_weight) + aux / 3.0
            loss_amr = F.binary_cross_entropy_with_logits(out1["amr_logits"], amr, pos_weight=amr_pos_weight) + aux / 3.0
            loss_exp = F.binary_cross_entropy_with_logits(out1["expansion_logits"], exp, pos_weight=exp_pos_weight) + aux / 3.0

            losses_tasks = [loss_mob, loss_amr, loss_exp]
            grads = []
            for k, lt in enumerate(losses_tasks):
                optimizer.zero_grad(set_to_none=True)
                lt.backward(retain_graph=(k < 2))
                g_vec = []
                for p in model.parameters():
                    if p.requires_grad:
                        if p.grad is not None:
                            g_vec.append(p.grad.view(-1))
                        else:
                            g_vec.append(torch.zeros(p.numel(), device=device))
                grads.append(torch.cat(g_vec))

            optimizer.zero_grad(set_to_none=True)
            proj_grads = [g.clone() for g in grads]
            for i in torch.randperm(3, device=device).tolist():
                for j in torch.randperm(3, device=device).tolist():
                    if i != j:
                        dot = torch.dot(proj_grads[i], grads[j])
                        if dot < 0:
                            proj_grads[i] -= (dot / (torch.dot(grads[j], grads[j]) + 1e-8)) * grads[j]

            sum_grad = sum(proj_grads)
            idx_p = 0
            for p in model.parameters():
                if p.requires_grad:
                    numel = p.numel()
                    p.grad = sum_grad[idx_p:idx_p + numel].view(p.shape).clone()
                    idx_p += numel

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append((loss_mob + loss_amr + loss_exp).item())
        scheduler.step()
        avg_loss = sum(losses) / len(losses)
        val_metrics = evaluate_kmer_transformer(model, device_val, device)
        row = {"epoch": epoch, "train_loss": avg_loss, **val_metrics}
        history.append(row)
        score = val_metrics.get("amr_auroc", 0) + val_metrics.get("expansion_auroc", 0) + val_metrics.get("mobility_balanced_accuracy", 0)
        if score > best_score:
            best_score = score
            patience_counter = 0
            model.save(artifact_dir / "kmer_transformer_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= config.get("patience", 5):
                break
    (artifact_dir / "kmer_transformer_history.json").write_text(json.dumps(history, indent=2))
    return artifact_dir / "kmer_transformer_best.pt", history


@torch.inference_mode()
def evaluate_kmer_transformer(model, data, device="cpu", batch_size: int = 32):
    model.eval()
    model.to(device)
    import numpy as np
    feats, specs, masks, sids = data["features"], data["spec_features"], data["masks"], data["scale_ids"]
    n = len(feats)
    mob_list, amr_list, exp_list = [], [], []
    for start in range(0, n, batch_size):
        end = start + batch_size
        out = model(feats[start:end].to(device), specs[start:end].to(device), masks[start:end].to(device), sids[start:end].to(device))
        mob_list.append(torch.softmax(out["mobility_logits"], dim=-1).cpu())
        amr_list.append(torch.sigmoid(out["amr_logits"]).cpu())
        exp_list.append(torch.sigmoid(out["expansion_logits"]).cpu())
    mob_p = torch.cat(mob_list, dim=0).numpy() if mob_list else np.zeros((0, 3))
    amr_p = torch.cat(amr_list, dim=0).numpy() if amr_list else np.zeros((0,))
    exp_p = torch.cat(exp_list, dim=0).numpy() if exp_list else np.zeros((0,))
    m = {}
    m.update(multiclass_metrics(data["mobility"].cpu().numpy(), mob_p, "mobility"))
    m.update(binary_metrics(data["amr"].cpu().numpy(), amr_p, "amr"))
    m.update(binary_metrics(data["expansion"].cpu().numpy(), exp_p, "expansion"))
    return m



