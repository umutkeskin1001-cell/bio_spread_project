from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dna_sentinel.metrics import binary_metrics, multiclass_metrics
from dna_sentinel.model import DnaSentinel, DnaSentinelConfig


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 8
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-3
    artifact_dir: str | Path = "artifacts/dna_sentinel"
    seed: int = 42
    device: str = "auto"
    weighted_loss: bool = True


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _loss(
    model: DnaSentinel,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    mobility_weight: torch.Tensor | None = None,
    amr_pos_weight: torch.Tensor | None = None,
    expansion_pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    tokens = batch["tokens"].to(device)
    mask = batch["mask"].to(device)
    mobility = batch["mobility"].to(device)
    amr = batch["amr"].to(device)
    expansion = batch["expansion"].to(device)
    out = model(tokens, mask)
    loss_mob = F.cross_entropy(out.mobility_logits, mobility, weight=mobility_weight)
    loss_amr = F.binary_cross_entropy_with_logits(out.amr_logits, amr, pos_weight=amr_pos_weight)
    loss_exp = F.binary_cross_entropy_with_logits(out.expansion_logits, expansion, pos_weight=expansion_pos_weight)
    entropy = -(out.evidence_weights.clamp_min(1e-8) * out.evidence_weights.clamp_min(1e-8).log()).sum(dim=1).mean()
    return loss_mob + loss_amr + loss_exp + 0.005 * entropy


def train_model(model: DnaSentinel, train_ds: Dataset, val_ds: Dataset, cfg: TrainConfig) -> tuple[Path, list[dict[str, float]]]:
    _seed(cfg.seed)
    device = _device(cfg.device)
    model.to(device)
    artifact_dir = Path(cfg.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    import os
    loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=min(4, os.cpu_count() or 1) if device.type == "cuda" else 0,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    weights = _task_weights(train_ds, device) if cfg.weighted_loss else (None, None, None)
    history: list[dict[str, float]] = []
    best_score = -1.0
    ckpt = artifact_dir / "best.pt"
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                loss = _loss(model, batch, device, *weights)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(loss.detach())
        metrics = evaluate(model, val_ds, cfg.batch_size, device=str(device))
        row = {"epoch": float(epoch), "train_loss": float(torch.stack([val.cpu() for val in losses]).mean().item()), **metrics}
        history.append(row)
        score = metrics.get("amr_auroc", 0.0) + metrics.get("expansion_auroc", 0.0) + metrics.get("mobility_balanced_accuracy", 0.0)
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_config": model.cfg.to_dict(),
                    "train_config": _safe_train_config(cfg),
                    "metrics": metrics,
                },
                ckpt,
            )
    (artifact_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return ckpt, history


@torch.inference_mode()
def evaluate(model: DnaSentinel, ds: Dataset, batch_size: int = 32, device: str = "auto") -> dict[str, float]:
    dev = _device(device)
    model.to(dev)
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, pin_memory=(dev.type == "cuda"))
    ys_mob, logits_mob, ys_amr, ps_amr, ys_exp, ps_exp = [], [], [], [], [], []
    for batch in loader:
        out = model(batch["tokens"].to(dev), batch["mask"].to(dev))
        ys_mob.extend(batch["mobility"].cpu().numpy().tolist())
        logits_mob.append(out.mobility_logits.detach().cpu().numpy())
        ys_amr.extend(batch["amr"].cpu().numpy().tolist())
        ps_amr.extend(torch.sigmoid(out.amr_logits).detach().cpu().numpy().tolist())
        ys_exp.extend(batch["expansion"].cpu().numpy().tolist())
        ps_exp.extend(torch.sigmoid(out.expansion_logits).detach().cpu().numpy().tolist())
    mob_arr = np.concatenate(logits_mob, axis=0) if logits_mob else np.zeros((0, 3))
    metrics = {}
    metrics.update(multiclass_metrics(ys_mob, mob_arr, "mobility"))
    metrics.update(binary_metrics(ys_amr, ps_amr, "amr"))
    metrics.update(binary_metrics(ys_exp, ps_exp, "expansion"))
    return metrics


def load_checkpoint(path: str | Path, device: str = "cpu") -> DnaSentinel:
    state = torch.load(path, map_location=device, weights_only=True)
    cfg = DnaSentinelConfig(**state["model_config"])
    model = DnaSentinel(cfg)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


def _safe_train_config(cfg: TrainConfig) -> dict:
    raw = asdict(cfg)
    raw["artifact_dir"] = str(raw["artifact_dir"])
    return raw


def _task_weights(ds: Dataset, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    records = getattr(ds, "records", [])
    mobility = np.asarray([r.mobility for r in records], dtype=int)
    amr = np.asarray([r.amr for r in records], dtype=float)
    expansion = np.asarray([r.expansion for r in records], dtype=float)
    counts = np.bincount(mobility, minlength=3).clip(min=1)
    mob_w = torch.tensor((counts.sum() / (3 * counts)), dtype=torch.float32, device=device)
    amr_w = torch.tensor([(len(amr) - amr.sum()) / max(1.0, amr.sum())], dtype=torch.float32, device=device)
    exp_w = torch.tensor([(len(expansion) - expansion.sum()) / max(1.0, expansion.sum())], dtype=torch.float32, device=device)
    return mob_w, amr_w, exp_w
