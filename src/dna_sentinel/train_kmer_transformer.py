"""KmerTransformer training and evaluation loop."""
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-5)
    window_dropout = WindowDropout(config.get("window_dropout", 0.25))
    artifact_dir = Path(config["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    mob_counts = torch.bincount(train_data["mobility"].long(), minlength=3).float().clamp(min=1)
    mob_weight = (mob_counts.sum() / (3 * mob_counts)).to(device)
    n_train = len(train_data["features"])
    amr_pos = float(train_data["amr"].sum().item() / max(1, n_train))
    exp_pos = float(train_data["expansion"].sum().item() / max(1, n_train))
    amr_pos_weight = torch.tensor([(1 - amr_pos) / max(amr_pos, 1e-6)], device=device)
    exp_pos_weight = torch.tensor([(1 - exp_pos) / max(exp_pos, 1e-6)], device=device)
    best_score = -1.0
    patience_counter = 0
    history = []
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        losses = []
        idx = torch.randperm(len(train_data["features"]))
        for start in range(0, len(idx), config["batch_size"]):
            bi = idx[start:start + config["batch_size"]]
            feat = train_data["features"][bi].to(device)
            spec = train_data["spec_features"][bi].to(device)
            mask = train_data["masks"][bi].to(device)
            sid = train_data["scale_ids"][bi].to(device)
            mob = train_data["mobility"][bi].to(device)
            amr = train_data["amr"][bi].to(device)
            exp = train_data["expansion"][bi].to(device)
            [feat1, spec1], mask1 = window_dropout([feat, spec], mask, training=True)
            out1 = model(feat1, spec1, mask1, sid)
            [feat2, spec2], mask2 = window_dropout([feat, spec], mask, training=True)
            out2 = model(feat2, spec2, mask2, sid)
            loss_task = (
                F.cross_entropy(out1["mobility_logits"], mob, weight=mob_weight)
                + F.binary_cross_entropy_with_logits(out1["amr_logits"], amr, pos_weight=amr_pos_weight)
                + F.binary_cross_entropy_with_logits(out1["expansion_logits"], exp, pos_weight=exp_pos_weight)
            )
            loss_cl = (1.0 - F.cosine_similarity(out1["pooled"], out2["pooled"], dim=-1)).mean()
            entropy = -(out1["evidence_weights"].clamp_min(1e-8) * out1["evidence_weights"].clamp_min(1e-8).log()).sum(dim=1).mean()
            loss = loss_task + 0.1 * loss_cl + 0.005 * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.detach())
        scheduler.step()
        avg_loss = torch.stack([val.cpu() for val in losses]).mean().item()
        val_metrics = evaluate_kmer_transformer(model, val_data, device)
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
def evaluate_kmer_transformer(model, data, device="cpu"):
    model.eval()
    model.to(device)
    out = model(data["features"].to(device), data["spec_features"].to(device), data["masks"].to(device), data["scale_ids"].to(device))
    mob_p = torch.softmax(out["mobility_logits"], dim=-1).cpu().numpy()
    amr_p = torch.sigmoid(out["amr_logits"]).cpu().numpy()
    exp_p = torch.sigmoid(out["expansion_logits"]).cpu().numpy()
    m = {}
    m.update(multiclass_metrics(data["mobility"].cpu().numpy(), mob_p, "mobility"))
    m.update(binary_metrics(data["amr"].cpu().numpy(), amr_p, "amr"))
    m.update(binary_metrics(data["expansion"].cpu().numpy(), exp_p, "expansion"))
    return m
