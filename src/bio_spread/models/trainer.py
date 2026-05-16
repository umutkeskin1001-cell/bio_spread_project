from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from bio_spread.models.components import (
    AdaptiveLossWeighting,
    CAGradProjector,
    PlattScaler,
)
from bio_spread.models.sovereign import BioSpreadModel, ModelOutput
from bio_spread.utils.metrics import bootstrap_metrics, classification_metrics, expected_calibration_error

logger = logging.getLogger(__name__)


def ranking_loss(probs: torch.Tensor, targets: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    diff = probs[:, 1:] - probs[:, :-1]
    target_diff = targets[:, 1:] - targets[:, :-1]
    valid = (target_diff > 0) & (targets[:, :-1] >= 0) & (targets[:, 1:] >= 0)
    if not valid.any():
        return torch.tensor(0.0, device=probs.device, requires_grad=True)
    return F.relu(margin - diff[valid]).mean()


def hazard_masked_bce(logits: torch.Tensor, targets: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    valid = targets >= 0
    if not valid.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    return F.binary_cross_entropy_with_logits(
        logits[valid],
        targets[valid].clamp(min=0),
        pos_weight=pos_weight,
    )


def cold_start_hard_negative_loss(
    cold_logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
    hard_ratio: float = 0.2,
    fn_penalty: float = 0.5,
) -> torch.Tensor:
    B = cold_logits.size(0)
    cold_probs = torch.sigmoid(cold_logits.detach())
    per_sample_loss = F.binary_cross_entropy_with_logits(
        cold_logits, targets, reduction="none", pos_weight=pos_weight,
    )
    if per_sample_loss.dim() > 1:
        per_sample_loss = per_sample_loss.mean(dim=-1)

    k = max(1, int(B * hard_ratio))
    hard_indices = torch.topk(per_sample_loss, min(k, B)).indices
    weights = torch.ones_like(per_sample_loss)
    weights[hard_indices] = 2.0

    fn_mask = (cold_probs < 0.5) & (targets == 1.0)
    fn_weight = fn_mask.float()
    if fn_weight.dim() > 1:
        fn_weight = fn_weight.mean(dim=-1)
    fn_weight = fn_weight * fn_penalty
    weights = weights + fn_weight

    return (per_sample_loss * weights).mean()


class BioSpreadTrainer:

    def __init__(
        self,
        model: BioSpreadModel,
        device: str = "cpu",
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        epochs: int = 50,
        patience: int = 10,
        warmup_epochs: int = 5,
        grad_clip: float = 1.0,
        lambda_count: float = 0.15,
        lambda_rank: float = 0.10,
        lambda_cold: float = 0.5,
        lambda_kd: float = 1.0,
        kd_temperature: float = 1.0,
        lambda_all: float = 1.0,
        lambda_gate: float = 0.05,
        lambda_edl: float = 1.0,
        lambda_phylo: float = 0.01,
        lambda_info_nce: float = 0.1,
        lambda_prior: float = 0.05,
        temporal_masking_prob: float = 0.3,
        gaussian_noise_std: float = 0.05,
        gate_entropy_target: float = 0.4,
        pos_weight: Optional[float] = None,
        calibrate: bool = True,
        calibrate_cold: bool = True,
        use_adaptive_loss: bool = False,
        use_hard_negative_mining: bool = False,
        use_curriculum: bool = False,
        use_cagrad: bool = False,
        cagrad_c: float = 0.4,
        phylo_smooth_tau: float = 0.5,
        phylo_smooth_k: int = 24,
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
        self.lambda_kd = lambda_kd
        self.kd_temperature = kd_temperature
        self.lambda_all = lambda_all
        self.lambda_gate = lambda_gate
        self.lambda_edl = lambda_edl
        self.lambda_phylo = lambda_phylo
        self.lambda_info_nce = lambda_info_nce
        self.lambda_prior = lambda_prior
        self.temporal_masking_prob = temporal_masking_prob
        self.gaussian_noise_std = gaussian_noise_std
        self.gate_entropy_target = gate_entropy_target
        self.calibrate = calibrate
        self.calibrate_cold = calibrate_cold
        self.use_adaptive_loss = use_adaptive_loss
        self.use_hard_negative_mining = use_hard_negative_mining
        self.use_curriculum = use_curriculum
        self.use_cagrad = use_cagrad
        self.cagrad_c = cagrad_c
        self.phylo_smooth_tau = phylo_smooth_tau
        self.phylo_smooth_k = phylo_smooth_k

        self.platt_scalers = nn.ModuleList([PlattScaler().to(device) for _ in range(3)])
        self.cold_platt_scalers = nn.ModuleList([PlattScaler().to(device) for _ in range(3)])

        self.base_lr = lr

        # Separate param groups for hyperbolic vs standard optimizers
        hyperbolic_params = []
        standard_params = []
        for name, param in model.named_parameters():
            if "embeddings" in name and hasattr(model, "use_hyperbolic") and model.use_hyperbolic:
                hyperbolic_params.append(param)
            else:
                standard_params.append(param)

        param_groups = [{"params": standard_params, "lr": lr, "weight_decay": weight_decay}]
        if hyperbolic_params:
            param_groups.append({"params": hyperbolic_params, "lr": lr, "weight_decay": 0.0})

        self.optimizer = optim.AdamW(param_groups)
        warmup_end = max(epochs - warmup_epochs, 1)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=warmup_end, eta_min=lr * 0.01)
        self.pos_weight = torch.full((3,), pos_weight or 1.0, device=device)
        self.loss_scale_factors = None
        self._zero = torch.tensor(0.0, device=self.device)
        self._step_count = 0

        self.adaptive_loss: AdaptiveLossWeighting | None = None
        if use_adaptive_loss:
            self.adaptive_loss = AdaptiveLossWeighting(n_losses=5).to(device)
            self.optimizer.add_param_group({"params": self.adaptive_loss.parameters(), "lr": lr * 0.1})

    def _get_curriculum_params(self, epoch: int) -> dict:
        """Simplified production curriculum."""
        if not self.use_curriculum:
            return {
                "lambda_proxy": 1.0,
                "lambda_contrast": 0.5,
                "lambda_kd": 0.5,
                "noise_std": self.gaussian_noise_std,
                "temporal_masking_prob": self.temporal_masking_prob,
            }

        # Early phase: Focus on backbone and proxy alignment
        if epoch <= 5:
            return {
                "lambda_proxy": 2.0,
                "lambda_contrast": 1.0,
                "lambda_kd": 0.1,
                "noise_std": 0.01,
                "temporal_masking_prob": 0.1,
            }
        # Mid phase: Introduce KD and noise
        elif epoch <= 15:
            return {
                "lambda_proxy": 1.0,
                "lambda_contrast": 0.5,
                "lambda_kd": 0.5,
                "noise_std": 0.03,
                "temporal_masking_prob": 0.2,
            }
        # Final phase: Refinement
        else:
            return {
                "lambda_proxy": 0.5,
                "lambda_contrast": 0.2,
                "lambda_kd": 1.0,
                "noise_std": 0.05,
                "temporal_masking_prob": 0.35,
            }

    def _get_adaptive_temporal_mask(
        self, seq_lens: torch.Tensor, epoch: int
    ) -> torch.Tensor:
        B = seq_lens.size(0)
        params = self._get_curriculum_params(epoch)
        base_prob = params["temporal_masking_prob"]

        if base_prob <= 0:
            return torch.zeros(B, dtype=torch.bool, device=self.device)

        history_weight = torch.clamp(1.0 / (seq_lens.float().sqrt() + 0.1), min=0.5, max=2.0)
        probs = base_prob * history_weight
        probs = torch.clamp(probs, min=0.05, max=0.9)
        mask = torch.rand(B, device=self.device) < probs
        return mask

    def _compute_all_losses(
        self,
        out: ModelOutput,
        batch: Dict[str, torch.Tensor],
        epoch: int,
    ) -> Dict[str, torch.Tensor]:
        targets = batch["hazard"]
        lengths = batch["seq_len"]
        temporal_mask = batch.get("temporal_mask")
        if temporal_mask is None:
            temporal_mask = torch.zeros(targets.size(0), dtype=torch.bool, device=targets.device)
        params = self._get_curriculum_params(epoch)

        B = targets.size(0)
        idx = (lengths - 1).clamp(min=0)
        last_targets = targets[range(B), idx]

        # 1. Main Hazard Loss (L_hazard)
        if self.model.use_evidential and out.alpha_pos is not None:
            loss_hazard = self._zero
            for h in range(3):
                valid = last_targets[:, h] >= 0
                if valid.any():
                    loss_hazard = loss_hazard + self.model.hazard_head.loss(
                        out.alpha_pos[valid, h], last_targets[valid, h], self.pos_weight[h:h+1]
                    )
        else:
            loss_hazard = self._zero
            for h in range(3):
                loss_hazard = loss_hazard + hazard_masked_bce(
                    out.hazard_logits[:, h], last_targets[:, h], self.pos_weight[h:h+1]
                )

        # 2. Proxy Alignment Loss (L_proxy): MSE(proxy, h_pooled.detach())
        loss_proxy = self._zero
        if out.proxy_temporal is not None and out.h_temporal is not None:
            # Manifold alignment: proxy should predict the temporal features
            # We detach h_temporal to avoid proxy affecting the temporal encoder directly
            loss_proxy = F.mse_loss(out.proxy_temporal, out.h_temporal.detach())

        # 3. Contrastive Alignment Loss (L_contrast): InfoNCE between proxy and actual
        loss_contrast = self._zero
        if out.proxy_temporal is not None and out.h_temporal is not None:
            from bio_spread.models.components import contrastive_loss
            # Contrastive alignment between proxy (learned from static) and temporal (actual sequence)
            loss_contrast = contrastive_loss(F.normalize(out.proxy_temporal, dim=-1), 
                                           F.normalize(out.h_temporal, dim=-1))

        # 4. Knowledge Distillation Loss (L_kd): Teacher (Hazard) -> Cold Path (Cold Logits)
        loss_kd = self._zero
        if out.cold_logits is not None:
            teacher_probs = torch.sigmoid(out.hazard_logits.detach())
            for h in range(3):
                valid = last_targets[:, h] >= 0
                if valid.any():
                    # We want cold path to mimic the full (temporal-aware) path
                    loss_kd = loss_kd + F.binary_cross_entropy_with_logits(
                        out.cold_logits[valid, h], teacher_probs[valid, h]
                    )

        # 5. Auxiliary losses (optional, kept for stability if needed)
        loss_count = self._zero
        if batch.get("count") is not None:
            count_targets = batch["count"]
            count_valid = count_targets >= 0
            if count_valid.any():
                loss_count = F.smooth_l1_loss(
                    out.count_logits[count_valid],
                    torch.log1p(count_targets[count_valid].clamp(min=0)),
                )

        # Combine with Uncertainty-Weighted Multi-Task or simple weights
        losses_dict = {
            "hazard": loss_hazard,
            "proxy": loss_proxy,
            "contrast": loss_contrast,
            "kd": loss_kd,
            "count": loss_count,
        }

        if self.use_adaptive_loss and self.adaptive_loss is not None:
            # Automatically balance losses using Kendall 2018 approach
            total = self.adaptive_loss(losses_dict)
        else:
            # Static weighting based on curriculum params
            total = (
                loss_hazard
                + params["lambda_proxy"] * loss_proxy
                + params["lambda_contrast"] * loss_contrast
                + params["lambda_kd"] * loss_kd
                + 0.1 * loss_count
            )

        losses_dict["total"] = total
        return losses_dict

    def _measure_loss_scales(self, loader: DataLoader, n_batches: int = 50) -> Dict[str, float]:
        self.model.train()
        accum = {
            "hazard": 0.0,
            "proxy": 0.0,
            "contrast": 0.0,
            "kd": 0.0,
            "count": 0.0,
        }
        counts = {k: 0 for k in accum}

        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= n_batches:
                    break

                static = batch["static"].to(self.device)
                seq = batch["seq"].to(self.device)
                mask = batch["mask"].to(self.device)
                lengths = batch["seq_len"].to(self.device)
                targets = batch["hazard"].to(self.device)
                counts_t = batch["count"].to(self.device)
                taxonomy_idxs = batch.get("taxonomy")
                if taxonomy_idxs is not None:
                    taxonomy_idxs = taxonomy_idxs.to(self.device)

                out = self.model(static, seq, mask, taxonomy_idxs, cat_inputs=batch.get("cat_inputs"))

                device_batch = {
                    "hazard": targets,
                    "count": counts_t,
                    "seq_len": lengths,
                    "mask": mask,
                }
                losses = self._compute_all_losses(out, device_batch, epoch=1)

                for k in accum:
                    val = losses[k].item()
                    accum[k] += val
                    counts[k] += 1

        scale_factors = {}
        for k in accum:
            mean_val = accum[k] / max(counts[k], 1)
            scale_factors[k] = 1.0 / max(mean_val, 1e-8)

        ref = scale_factors.get("hazard", 1.0)
        for k in scale_factors:
            scale_factors[k] = scale_factors[k] / ref

        return scale_factors

    def _compute_pos_weight(self, loader: DataLoader) -> torch.Tensor:
        total = torch.zeros(3, device=self.device)
        pos = torch.zeros(3, device=self.device)
        for batch in loader:
            targets = batch["hazard"]
            mask = batch["mask"]
            valid = (targets >= 0) & mask.unsqueeze(-1).bool()
            for h in range(3):
                pos_h = (targets[..., h][valid[..., h]] > 0).sum()
                total_h = valid[..., h].sum()
                pos[h] += pos_h
                total[h] += total_h
        total = total.clamp(min=1)
        pos = pos.clamp(min=1)
        neg = total - pos
        return (neg / pos).clamp(max=100.0)

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        self.model.train()
        loss_total = 0.0
        n_masked = 0
        skipped_batches = 0
        params = self._get_curriculum_params(epoch)

        if epoch <= self.warmup_epochs:
            lr_scale = 0.1 + 0.9 * (epoch - 1) / max(self.warmup_epochs, 1)
            for pg in self.optimizer.param_groups:
                if pg["lr"] > 0:
                    pg["lr"] = self.base_lr * lr_scale

        for batch in train_loader:
            self._step_count += 1
            static = batch["static"].to(self.device)
            seq = batch["seq"].to(self.device)
            mask = batch["mask"].to(self.device)
            lengths = batch["seq_len"].to(self.device)
            targets = batch["hazard"].to(self.device)
            count_targets = batch["count"].to(self.device)
            taxonomy_idxs = batch.get("taxonomy")
            if taxonomy_idxs is not None:
                taxonomy_idxs = taxonomy_idxs.to(self.device)

            seq.shape[0]

            seq_noised = seq
            if params["noise_std"] > 0:
                noise = torch.randn_like(seq) * params["noise_std"]
                seq_noised = seq + noise * mask.unsqueeze(-1)

            temporal_mask = self._get_adaptive_temporal_mask(lengths, epoch)
            n_masked += temporal_mask.sum().item()

            # Conditional CAGrad: every 5 steps or when loss ratio > 2
            do_cagrad = self.use_cagrad and (self._step_count % 5 == 0)

            out = self.model(
                static, seq_noised, mask, taxonomy_idxs, temporal_mask=temporal_mask,
                cat_inputs=batch.get("cat_inputs"),
            )

            device_batch = {
                "hazard": targets,
                "count": count_targets,
                "seq_len": lengths,
                "mask": mask,
                "temporal_mask": temporal_mask,
            }

            if do_cagrad:
                # Compute individual losses first to check ratio
                with torch.no_grad():
                    ratio_check = self._compute_all_losses(out, device_batch, epoch)
                    loss_vals = {k: v.item() for k, v in ratio_check.items()
                                 if k not in ("total",) and torch.isfinite(v)}
                    if loss_vals:
                        max_l = max(loss_vals.values())
                        min_l = min(loss_vals.values())
                        if not (max_l > 2.0 * min_l and min_l > 1e-8):
                            do_cagrad = False

            if do_cagrad:
                self.optimizer.zero_grad(set_to_none=True)
                losses = self._compute_all_losses(out, device_batch, epoch)
                loss = losses["total"]
                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                    loss_total += loss.item()
                else:
                    skipped_batches += 1
                    self.optimizer.zero_grad(set_to_none=True)
            else:
                losses = self._compute_all_losses(out, device_batch, epoch)
                loss = losses["total"]

                if not torch.isfinite(loss):
                    skipped_batches += 1
                    logger.warning("NaN/Inf loss encountered -- skipping batch")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                loss_total += loss.item()

            # Update retriever prototypes with cold input from this batch
            if self.model.use_retrieval and hasattr(self.model, "retriever"):
                cold_input = getattr(out, "cold_input", None)
                if cold_input is not None:
                    tgt_idx = (lengths - 1).clamp(min=0)
                    retriever_targets = targets[range(targets.size(0)), tgt_idx]
                    self.model.retriever.update(cold_input, retriever_targets)

        if epoch >= self.warmup_epochs:
            self.scheduler.step()

        avg_loss = loss_total / max(len(train_loader), 1)
        lr_now = self.optimizer.param_groups[0]["lr"]
        mask_pct = 100.0 * n_masked / max(len(train_loader.dataset), 1)
        logger.info(
            "Epoch %3d | Loss: %.4f | Mask: %.0f%% | LR: %.2e",
            epoch, avg_loss, mask_pct, lr_now,
        )

        if skipped_batches > 0:
            logger.warning("Epoch %d: Skipped %d batches due to NaN/Inf loss", epoch, skipped_batches)

        return avg_loss

    def _should_stop(self) -> bool:
        return self._patience_counter >= self.patience

    def _save_checkpoint(self, metrics: Dict) -> None:
        torch.save(self.model.state_dict(), self._artifact_dir / "best_model.pt")
        with open(self._artifact_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        cal_loader: Optional[DataLoader] = None,
        cold_cal_loader: Optional[DataLoader] = None,
    ) -> Path:
        if len(train_loader) == 0:
            raise ValueError("Empty training loader")

        self.loss_scale_factors = None

        self.pos_weight = self._compute_pos_weight(train_loader)
        self.loss_scale_factors = self._measure_loss_scales(train_loader)
        logger.info("Loss scale factors: %s", self.loss_scale_factors)
        run_id = "BS_%s" % datetime.now().strftime("%Y%m%d_%H%M%S")
        self._artifact_dir = Path("artifacts") / run_id
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        best_auc = -1.0
        self._patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            self._train_epoch(train_loader, epoch)

            if val_loader is not None:
                metrics = self.evaluate(val_loader)
                auc_h1 = metrics.get("roc_auc_h1", 0.0)
                auc_h2 = metrics.get("roc_auc_h2", 0.0)
                auc_h3 = metrics.get("roc_auc_h3", 0.0)
                val_auc = 0.5 * auc_h3 + 0.25 * auc_h1 + 0.25 * auc_h2
                logger.info(
                    "  Val | Composite: %.4f | AUC(h1): %.4f "
                    "AUC(h2): %.4f "
                    "AUC(h3): %.4f | "
                    "F1: %.4f | ECE: %.4f",
                    val_auc, auc_h1, auc_h2, auc_h3,
                    metrics.get("f1", 0.0), metrics.get("ece", 0.0),
                )
                if val_auc > best_auc:
                    best_auc = val_auc
                    self._patience_counter = 0
                    self._save_checkpoint(metrics)
                    logger.info("  -> New best (AUC: %.4f)", best_auc)
                else:
                    self._patience_counter += 1
                    if self._should_stop():
                        logger.info("Early stop at epoch %d", epoch)
                        break

        best_path = self._artifact_dir / "best_model.pt"
        if best_path.exists():
            self.model.load_state_dict(torch.load(best_path, map_location=self.device, weights_only=True))

        cal_data = cal_loader if cal_loader is not None else val_loader
        if self.calibrate and cal_data is not None:
            self._learn_platt(cal_data, self.platt_scalers, force_temporal_mask=False)
            torch.save(
                {f"scaler_h{h}": self.platt_scalers[h].state_dict() for h in range(3)},
                self._artifact_dir / "platt.pt",
            )

        cold_cal = cold_cal_loader if cold_cal_loader is not None else cal_data
        if self.calibrate_cold and cold_cal is not None:
            self._learn_platt(cold_cal, self.cold_platt_scalers, force_temporal_mask=True)
            torch.save(
                {f"scaler_h{h}": self.cold_platt_scalers[h].state_dict() for h in range(3)},
                self._artifact_dir / "platt_cold.pt",
            )

        return self._artifact_dir

    def _learn_platt(self, loader: DataLoader, scalers: nn.ModuleList,
                     force_temporal_mask: Optional[bool] = None) -> None:
        self.model.eval()
        for scaler in scalers:
            scaler.train()

        all_logits = {h: [] for h in range(3)}
        all_targets = {h: [] for h in range(3)}
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

                temporal_mask = torch.ones(static.size(0), dtype=torch.bool, device=self.device) if force_temporal_mask else None

                out = self.model(static, seq, mask, taxonomy_idxs, temporal_mask=temporal_mask,
                                 cat_inputs=batch.get("cat_inputs"))
                idx = (lengths - 1).clamp(min=0)
                last_logits = out.hazard_logits
                last_targets = targets[range(static.size(0)), idx]
                valid = last_targets >= 0
                for h in range(3):
                    v = valid[:, h]
                    if v.any():
                        all_logits[h].append(last_logits[v, h])
                        all_targets[h].append(last_targets[v, h])

        for h in range(3):
            if not all_logits[h]:
                logger.warning("No valid samples for Platt calibration (horizon %d)", h + 1)
                continue

            logits = torch.cat(all_logits[h])
            targets = torch.cat(all_targets[h])
            scaler = scalers[h]

            opt = optim.SGD(scaler.parameters(), lr=0.1)
            for _ in range(100):
                opt.zero_grad()
                scaled = scaler(logits)
                loss = F.binary_cross_entropy_with_logits(scaled, targets)
                loss.backward()
                opt.step()

            a_val = float(scaler.a.cpu().item())
            b_val = float(scaler.b.cpu().item())
            logger.info("Platt scaling (h%d): a=%.3f, b=%.3f", h + 1, a_val, b_val)

    def _calibrated_probs(self, logits: torch.Tensor, use_cold_scaler: bool = False) -> torch.Tensor:
        scalers = self.cold_platt_scalers if use_cold_scaler else self.platt_scalers
        probs = torch.zeros_like(logits)
        for h in range(3):
            scaled = scalers[h](logits[:, h])
            probs[:, h] = torch.sigmoid(scaled)
        return probs

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, use_cold_scaler: bool = False,
                 force_temporal_mask: bool = False) -> Dict:
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

            temporal_mask = torch.ones(static.size(0), dtype=torch.bool, device=self.device) if force_temporal_mask else None
            out = self.model(static, seq, mask, taxonomy_idxs, temporal_mask=temporal_mask,
                             cat_inputs=batch.get("cat_inputs"))

            logits = out.hazard_logits
            probs = self._calibrated_probs(logits, use_cold_scaler=use_cold_scaler)

            B = static.size(0)
            idx = (lengths - 1).clamp(min=0)
            for h in range(3):
                prob_h = probs[:, h].cpu().numpy()
                target_h = targets[range(B), idx, h].cpu().numpy()
                valid = target_h >= 0
                if valid.any():
                    all_probs[h].append(prob_h[valid])
                    all_targets[h].append(target_h[valid])

        if not any(len(v) > 0 for v in all_probs.values()):
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

        if all_probs[2]:
            y_prob = np.concatenate(all_probs[2])
            y_true = np.concatenate(all_targets[2])
            if len(np.unique(y_true)) > 1:
                cls_metrics = classification_metrics(y_true, y_prob)
                metrics.update(cls_metrics)
                metrics["ece"] = expected_calibration_error(y_true, y_prob)
                ci = bootstrap_metrics(y_true, y_prob)
                if ci:
                    metrics["roc_auc_ci_low"] = ci.get("ci_low", 0.0)
                    metrics["roc_auc_ci_high"] = ci.get("ci_high", 0.0)
            else:
                metrics["roc_auc"] = 0.5
                metrics["f1"] = 0.0
                metrics["ece"] = 0.0
                metrics["positive_rate"] = float(y_true.mean())
            metrics["n"] = int(len(y_true))
        else:
            metrics.setdefault("roc_auc", 0.5)
            metrics.setdefault("f1", 0.0)
            metrics.setdefault("ece", 0.0)
            metrics["n"] = 0

        return metrics

    @torch.no_grad()
    def calibrate_aci(self, loader: DataLoader, alpha: float = 0.1):
        """v3+ Adaptive Conformal Inference (ACI) calibration."""
        self.model.eval()
        all_residuals = []
        for batch in loader:
            static = batch["static"].to(self.device)
            seq = batch["seq"].to(self.device)
            mask = batch["mask"].to(self.device)
            targets = batch["hazard"].to(self.device)
            lengths = batch["seq_len"].to(self.device)
            
            out = self.model(static, seq, mask, taxonomy_idxs=batch.get("taxonomy"))
            idx = (lengths - 1).clamp(min=0)
            probs = torch.sigmoid(out.hazard_logits)
            tgt = targets[range(static.size(0)), idx]
            
            valid = tgt >= 0
            if valid.any():
                res = torch.abs(probs[valid] - tgt[valid])
                all_residuals.append(res.cpu().numpy())
        
        if not all_residuals:
            return 0.5
            
        residuals = np.concatenate(all_residuals)
        q_val = np.percentile(residuals, 100 * (1 - alpha))
        self.aci_quantile = q_val
        logger.info("ACI Calibration completed: Quantile(%.2f) = %.4f", 1-alpha, q_val)
        return q_val

    def _apply_aci(self, probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (probs, prediction_set_size/confidence_interval)."""
        q = getattr(self, "aci_quantile", 0.5)
        lower = (probs - q).clamp(0, 1)
        upper = (probs + q).clamp(0, 1)
        return lower, upper

    @torch.no_grad()
    def predict_frontier(self, batch: dict) -> dict:
        """
        v3+ Nihai Frontier Prediction.
        Returns: point estimate + [p_low, p_high] + uncertainty_band + routing_decision
        """
        self.model.eval()
        static = batch["static"].to(self.device)
        seq = batch["seq"].to(self.device)
        mask = batch["mask"].to(self.device)
        taxonomy_idxs = batch.get("taxonomy")
        if taxonomy_idxs is not None:
            taxonomy_idxs = taxonomy_idxs.to(self.device)
            
        out = self.model(static, seq, mask, taxonomy_idxs, cat_inputs=batch.get("cat_inputs"))
        
        probs = torch.sigmoid(out.hazard_logits)
        p_low, p_high = self._apply_aci(probs)
        
        uncertainty = out.epistemic_var.mean(dim=-1) if out.epistemic_var is not None else torch.zeros(static.size(0))
        routing = out.routing_weight if out.routing_weight is not None else torch.zeros(static.size(0))
        
        return {
            "hazard_probs": probs,
            "p_interval_low": p_low,
            "p_interval_high": p_high,
            "uncertainty_band": uncertainty,
            "routing_decision": routing,
            "count_estimate": torch.expm1(out.count_logits),
            "is_cold_start": (routing > 0.5), # Heuristic decision
        }
