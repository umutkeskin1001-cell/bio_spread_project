"""
Sovereign-X Pro: Trainer with per-timestep BCE + temporal masking + cold-start aux.

Key innovation — Multi-snapshot training:
    Instead of computing BCE only on the LAST snapshot's hazard targets,
    we compute BCE on ALL valid snapshots' targets using per-timestep
    hazard predictions. This multiplies effective training data by ~3-8x.

Loss = BCE(hazard_last_snapshot)
     + lambda_all * BCE(hazard_all_timesteps)     ← NEW: trains on ALL snapshots
     + lambda_count * smooth_l1(count)
     + lambda_cold * BCE(cold_start_head)
     + lambda_rank * ranking(hazard_1,2,3)         ← FIXED: uses all 3 horizons
     + gate_entropy_penalty                        ← config-driven weight
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from bio_spread_reborn.models.components import PlattScaler
from bio_spread_reborn.models.sovereign import SovereignX
from bio_spread_reborn.utils.metrics import classification_metrics, expected_calibration_error

logger = logging.getLogger(__name__)


def ranking_loss(probs: torch.Tensor, targets: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    """Temporal ranking: later snapshot hazard >= earlier when target increases.

    Args:
        probs: (B, L) predicted hazard probabilities for ONE hazard index
        targets: (B, L) ground-truth targets for ONE hazard index
    Returns:
        scalar loss
    """
    B, L = probs.shape[:2]
    losses = []
    for b in range(B):
        t = targets[b]
        p = probs[b]
        for i in range(L - 1):
            # Only penalize if target goes UP and is valid
            if t[i + 1] > t[i] and t[i] >= 0 and t[i + 1] >= 0:
                losses.append(F.relu(margin - (p[i + 1] - p[i])))
    if not losses:
        return torch.tensor(0.0, device=probs.device, requires_grad=True)
    return torch.stack(losses).mean()


def hazard_masked_bce(logits: torch.Tensor, targets: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    """BCE with masking for -1 (censored) targets.

    Args:
        logits: (N, ...) raw logits
        targets: (N, ...) targets with -1 for censored
        pos_weight: weight for positive class
    Returns:
        scalar BCE loss over valid positions
    """
    valid = targets >= 0
    if not valid.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    return F.binary_cross_entropy_with_logits(
        logits[valid],
        targets[valid].clamp(min=0),
        pos_weight=pos_weight,
    )


class SovereignXTrainer:
    """Trainer for Sovereign-X Pro with multi-snapshot hazard loss.

    Config-driven parameters (all accessible via constructor):
        - gate_entropy_weight: weight for gate entropy penalty
        - gaussian_noise_std: std of noise added to temporal features
    """

    def __init__(
        self,
        model: SovereignX,
        device: str = "cpu",
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        epochs: int = 50,
        patience: int = 10,
        warmup_epochs: int = 5,
        grad_clip: float = 1.0,
        lambda_count: float = 0.15,
        lambda_rank: float = 0.10,
        lambda_cold: float = 0.25,
        lambda_all: float = 1.0,  # weight for per-timestep (all snapshots) loss
        lambda_gate: float = 0.05,  # weight for gate entropy penalty
        temporal_masking_prob: float = 0.3,
        gaussian_noise_std: float = 0.05,  # noise on temporal features
        gate_entropy_target: float = 0.4,  # min entropy before penalty kicks in
        pos_weight: Optional[float] = None,
        calibrate: bool = True,
        calibrate_cold: bool = True,
    ):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.warmup_epochs = warmup_epochs
        self.grad_clip = grad_clip
        self.lambda_count = lambda_count
        self.lambda_rank = lambda_rank
        self.lambda_cold = lambda_cold
        self.lambda_all = lambda_all
        self.lambda_gate = lambda_gate
        self.temporal_masking_prob = temporal_masking_prob
        self.gaussian_noise_std = gaussian_noise_std
        self.gate_entropy_target = gate_entropy_target
        self.calibrate = calibrate
        self.calibrate_cold = calibrate_cold

        self.platt_scaler = PlattScaler().to(device)
        self.cold_platt_scaler = PlattScaler().to(device)

        self.base_lr = lr
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=lr * 0.01)
        self.pos_weight = torch.tensor(pos_weight or 1.0, device=device)

    def _compute_pos_weight(self, loader: DataLoader) -> torch.Tensor:
        """Compute pos_weight from ALL non-censored targets (all timesteps, all horizons)."""
        total, pos = 0, 0
        for batch in loader:
            targets = batch["hazard"]  # (B, L, 3)
            mask = batch["mask"]  # (B, L)
            valid = (targets >= 0) & mask.unsqueeze(-1).bool()
            pos += (targets[valid] > 0.5).sum().item()
            total += valid.sum().item()
        if total == 0 or pos == 0:
            return torch.tensor(1.0, device=self.device)
        return torch.tensor((total - pos) / max(pos, 1), device=self.device)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        cal_loader: Optional[DataLoader] = None,
        cold_cal_loader: Optional[DataLoader] = None,
    ) -> Path:
        """Train model with multi-snapshot hazard loss.

        Args:
            train_loader: training data
            val_loader: early-stopping validation
            cal_loader: held-out data for temporal Platt scaling
            cold_cal_loader: held-out data for cold-start Platt scaling
        """
        self.pos_weight = self._compute_pos_weight(train_loader)
        run_id = f"SX_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        artifact_dir = Path("artifacts") / run_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        best_auc, patience_counter = -1.0, 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            loss_total = 0.0
            n_masked = 0

            # Linear warmup
            if epoch <= self.warmup_epochs:
                lr_scale = 0.1 + 0.9 * (epoch - 1) / max(self.warmup_epochs, 1)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.base_lr * lr_scale

            for batch in train_loader:
                static = batch["static"].to(self.device)
                seq = batch["seq"].to(self.device)
                mask = batch["mask"].to(self.device)
                lengths = batch["seq_len"].to(self.device)
                targets = batch["hazard"].to(self.device)
                counts = batch["count"].to(self.device)
                taxonomy_idxs = batch.get("taxonomy")
                if taxonomy_idxs is not None:
                    taxonomy_idxs = taxonomy_idxs.to(self.device)

                B, L = seq.shape[:2]

                # === Gaussian noise on temporal features (regularization) ===
                seq_noised = seq
                if self.gaussian_noise_std > 0:
                    noise = torch.randn_like(seq) * self.gaussian_noise_std
                    seq_noised = seq + noise * mask.unsqueeze(-1)

                # === Temporal masking for cold-start robustness ===
                temporal_mask = None
                if self.temporal_masking_prob > 0:
                    temporal_mask = torch.rand(B, device=self.device) < self.temporal_masking_prob
                    n_masked += temporal_mask.sum().item()

                # Forward pass
                hazard_logits, hazard_logits_all, count_pred, cold_logits, fused, weights, mask_out = self.model(
                    static, seq_noised, mask, taxonomy_idxs, temporal_mask=temporal_mask
                )

                idx = (lengths - 1).clamp(min=0)

                # ================================================================
                # LOSS 1: Final snapshot BCE (all 3 hazard horizons)
                # ================================================================
                last_logits = hazard_logits  # (B, 3)
                last_targets = targets[range(B), idx]  # (B, 3)
                loss_final = hazard_masked_bce(last_logits, last_targets, self.pos_weight)

                # ================================================================
                # LOSS 2: Per-timestep BCE — trains on ALL valid snapshots (GENIUS)
                # ================================================================
                # Mask out padded timesteps + fully-censored timesteps
                pad_free = mask.unsqueeze(-1).expand(-1, -1, 3).bool()  # (B, L, 3)
                loss_all = hazard_masked_bce(
                    hazard_logits_all[pad_free],
                    targets[pad_free],
                    self.pos_weight,
                )

                # ================================================================
                # LOSS 3: Cold-start auxiliary (static-only predictions)
                # ================================================================
                loss_cold = torch.tensor(0.0, device=self.device)
                if cold_logits is not None and temporal_mask is not None and temporal_mask.any():
                    cold_valid = temporal_mask & (last_targets >= 0).any(dim=1)
                    if cold_valid.any():
                        loss_cold = F.binary_cross_entropy_with_logits(
                            cold_logits[cold_valid],
                            last_targets[cold_valid].clamp(min=0),
                            pos_weight=self.pos_weight,
                        )

                # ================================================================
                # LOSS 4: Gate entropy regularization (prevents expert collapse)
                # ================================================================
                gate_entropy = -(weights * torch.log(weights.clamp(min=1e-8))).sum(dim=1).mean()
                loss_gate = F.relu(self.gate_entropy_target - gate_entropy) * self.lambda_gate

                # ================================================================
                # LOSS 5: Count prediction
                # ================================================================
                count_valid = counts >= 0
                loss_count = 0.0
                if count_valid.any():
                    loss_count = F.smooth_l1_loss(
                        count_pred[count_valid],
                        torch.log1p(counts[count_valid].clamp(min=0)),
                        reduction="mean",
                    )

                # ================================================================
                # LOSS 6: Ranking loss — within-sample temporal monotonicity
                # Uses ranking_loss() which compares consecutive timesteps
                # within each sample, NOT across different backbones
                # ================================================================
                probs_all = torch.sigmoid(hazard_logits_all)  # (B, L, 3)
                loss_rank = torch.tensor(0.0, device=self.device)
                n_rank = 0
                for h in range(3):
                    # ranking_loss expects (B, L) — will only use valid pairs
                    if targets[..., h].numel() > 0:
                        rl = ranking_loss(probs_all[..., h], targets[..., h])
                        loss_rank = loss_rank + rl
                        n_rank += 1
                if n_rank > 0:
                    loss_rank = loss_rank / n_rank

                # ================================================================
                # TOTAL LOSS
                # ================================================================
                loss = (
                    loss_final
                    + self.lambda_all * loss_all
                    + self.lambda_count * loss_count
                    + self.lambda_rank * loss_rank
                    + self.lambda_cold * loss_cold
                    + loss_gate
                )

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning("NaN/Inf loss encountered — skipping batch")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                loss_total += loss.item()

            self.scheduler.step()
            avg_loss = loss_total / max(len(train_loader), 1)
            lr_now = self.optimizer.param_groups[0]["lr"]
            mask_pct = 100.0 * n_masked / max(len(train_loader.dataset), 1)
            logger.info(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Mask: {mask_pct:.0f}% | LR: {lr_now:.2e}")

            if val_loader is not None:
                metrics = self.evaluate(val_loader)
                val_auc = metrics.get("roc_auc", 0.0)
                logger.info(
                    f"  Val | ROC AUC: {val_auc:.4f} | AUC(h1): {metrics.get('roc_auc_h1', 0.0):.4f} "
                    f"AUC(h2): {metrics.get('roc_auc_h2', 0.0):.4f} "
                    f"AUC(h3): {metrics.get('roc_auc_h3', 0.0):.4f} | "
                    f"F1: {metrics.get('f1', 0.0):.4f} | ECE: {metrics.get('ece', 0.0):.4f}"
                )
                if val_auc > best_auc:
                    best_auc = val_auc
                    patience_counter = 0
                    torch.save(self.model.state_dict(), artifact_dir / "best_model.pt")
                    with open(artifact_dir / "metrics.json", "w") as f:
                        json.dump(metrics, f, indent=2, default=str)
                    logger.info(f"  → New best (AUC: {best_auc:.4f})")
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(f"Early stop at epoch {epoch}")
                        break

        best_path = artifact_dir / "best_model.pt"
        if best_path.exists():
            self.model.load_state_dict(torch.load(best_path, map_location=self.device, weights_only=True))

        # Platt calibration (merged: same method, different scaler + optional temporal mask)
        cal_data = cal_loader if cal_loader is not None else val_loader
        if self.calibrate and cal_data is not None:
            self._learn_platt(cal_data, self.platt_scaler, force_temporal_mask=False)
            a_val = float(self.platt_scaler.a.cpu().item())
            b_val = float(self.platt_scaler.b.cpu().item())
            logger.info(f"Platt scaling: a={a_val:.3f}, b={b_val:.3f}")
            torch.save(self.platt_scaler.state_dict(), artifact_dir / "platt.pt")

        cold_cal = cold_cal_loader if cold_cal_loader is not None else cal_data
        if self.calibrate_cold and cold_cal is not None:
            self._learn_platt(cold_cal, self.cold_platt_scaler, force_temporal_mask=True)
            ca_val = float(self.cold_platt_scaler.a.cpu().item())
            cb_val = float(self.cold_platt_scaler.b.cpu().item())
            logger.info(f"Cold Platt scaling: a={ca_val:.3f}, b={cb_val:.3f}")
            torch.save(self.cold_platt_scaler.state_dict(), artifact_dir / "platt_cold.pt")

        return artifact_dir

    def _learn_platt(self, loader: DataLoader, scaler: PlattScaler, max_steps: int = 500,
                     force_temporal_mask: bool = False) -> None:
        """Learn Platt scaling (a, b) on a calibration set via L-BFGS.

        Args:
            loader: calibration data.
            scaler: PlattScaler instance to train.
            max_steps: max L-BFGS iterations.
            force_temporal_mask: if True, masks temporal features for cold-start calibration.
        """
        self.model.eval()
        scaler.train()
        opt = optim.LBFGS(scaler.parameters(), lr=0.01, max_iter=50)

        all_logits, all_targets = [], []
        with torch.no_grad():
            for batch in loader:
                static = batch["static"].to(self.device)
                seq = batch["seq"].to(self.device)
                mask = batch["mask"].to(self.device)
                lengths = batch["seq_len"].to(self.device)
                targets = batch["hazard"].to(self.device)
                taxonomy_idxs = batch.get("taxonomy")
                if taxonomy_idxs is not None:
                    taxonomy_idxs = taxonomy_idxs.to(self.device)

                B = static.size(0)
                temporal_mask = torch.ones(B, dtype=torch.bool, device=self.device) if force_temporal_mask else None

                out = self.model(static, seq, mask, taxonomy_idxs, temporal_mask=temporal_mask)
                idx = (lengths - 1).clamp(min=0)
                last_logits = out.hazard_logits[:, 2]
                last_targets = targets[range(B), idx, 2]
                valid = last_targets >= 0
                if valid.any():
                    all_logits.append(last_logits[valid])
                    all_targets.append(last_targets[valid])

        if not all_logits:
            logger.warning("No valid samples for Platt calibration")
            return

        logits = torch.cat(all_logits)
        targets = torch.cat(all_targets)

        def closure():
            opt.zero_grad()
            scaled = scaler(logits)
            loss = F.binary_cross_entropy_with_logits(scaled, targets)
            loss.backward()
            return loss

        for _ in range(max_steps):
            loss = opt.step(closure)
            if loss is None:
                break

    def _calibrated_probs(self, logits: torch.Tensor, use_cold_scaler: bool = False) -> torch.Tensor:
        """Apply Platt scaling to h3 logits (scaler is trained on h3 only).

        Returns all 3 probs with scaling applied only to the third horizon.
        """
        scaler = self.cold_platt_scaler if use_cold_scaler else self.platt_scaler
        B, H = logits.shape
        if H < 3:
            return torch.sigmoid(logits)
        scaled = logits.clone()
        scaled[:, 2] = scaler(scaled[:, 2:3]).squeeze(-1)
        return torch.sigmoid(scaled)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, use_cold_scaler: bool = False) -> Dict:
        """Evaluate on ALL 3 hazard horizons separately + aggregate metrics.

        Reports:
            - roc_auc, f1, ece, etc. (hazard_3 aggregate — backward compat)
            - roc_auc_h1, roc_auc_h2, roc_auc_h3 (per-horizon)
        """
        self.model.eval()
        all_probs = {h: [] for h in range(3)}
        all_targets = {h: [] for h in range(3)}

        for batch in loader:
            static = batch["static"].to(self.device)
            seq = batch["seq"].to(self.device)
            mask = batch["mask"].to(self.device)
            lengths = batch["seq_len"].to(self.device)
            targets = batch["hazard"].to(self.device)
            taxonomy_idxs = batch.get("taxonomy")
            if taxonomy_idxs is not None:
                taxonomy_idxs = taxonomy_idxs.to(self.device)

            out = self.model(static, seq, mask, taxonomy_idxs)
            probs = self._calibrated_probs(out.hazard_logits, use_cold_scaler=use_cold_scaler)

            idx = (lengths - 1).clamp(min=0)
            for h in range(3):
                last_h = probs[range(static.size(0)), h].cpu().numpy()
                last_t = targets[range(static.size(0)), idx, h].cpu().numpy()
                valid = last_t >= 0
                if valid.any():
                    all_probs[h].append(last_h[valid])
                    all_targets[h].append(last_t[valid])

        if not all(ap for ap in all_probs.values()):
            return {"roc_auc": 0.5, "f1": 0.0, "ece": 0.0, "n": 0}

        metrics = {}
        for h in range(3):
            if all_probs[h]:
                y_prob = np.concatenate(all_probs[h])
                y_true = np.concatenate(all_targets[h])
                if len(np.unique(y_true)) > 1:
                    m = classification_metrics(y_true, y_prob)
                    metrics[f"roc_auc_h{h + 1}"] = m["roc_auc"]
                    metrics[f"pr_auc_h{h + 1}"] = m["pr_auc"]
                    metrics[f"ece_h{h + 1}"] = expected_calibration_error(y_true, y_prob)
                else:
                    metrics[f"roc_auc_h{h + 1}"] = 0.5
                    metrics[f"pr_auc_h{h + 1}"] = float(y_true.mean())
                    metrics[f"ece_h{h + 1}"] = 0.0

        # Aggregate: use hazard_3 as primary metric (backward compat)
        if all_probs[2]:
            y_prob = np.concatenate(all_probs[2])
            y_true = np.concatenate(all_targets[2])
            if len(np.unique(y_true)) > 1:
                cls_metrics = classification_metrics(y_true, y_prob)
                metrics.update(cls_metrics)
                metrics["ece"] = expected_calibration_error(y_true, y_prob)
            else:
                metrics["roc_auc"] = 0.5
                metrics["f1"] = 0.0
                metrics["ece"] = 0.0
                metrics["positive_rate"] = float(y_true.mean())
            metrics["n"] = int(len(y_true))

        return metrics
