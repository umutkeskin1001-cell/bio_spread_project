from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bio_spread.data.dataset import (
    SequenceBatchSampler, SequenceDataset, load_normalizers, sequence_collate,
)
from bio_spread.data.loader import get_device
from bio_spread.data.snapshot import load_taxonomy_vocab
from bio_spread.models import create_model
from bio_spread.models.components import PlattScaler
from bio_spread.models.sovereign import BioSpreadModel
from bio_spread.models.trainer import BioSpreadTrainer
from bio_spread.utils.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_distill")


class DistillationTrainer:
    def __init__(
        self, teacher: BioSpreadModel, student: BioSpreadModel, device: str,
        lr: float = 3e-4, weight_decay: float = 1e-2, temp: float = 3.0, alpha: float = 0.7,
    ):
        self.teacher = teacher.to(device).eval()
        self.student = student.to(device).train()
        self.device = device
        self.temp = temp
        self.alpha = alpha
        self.optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)

    def train_epoch(self, loader: DataLoader) -> float:
        total_loss = 0.0
        for batch in loader:
            static = batch["static"].to(self.device)
            seq = batch["seq"].to(self.device)
            mask = batch["mask"].to(self.device)
            targets = batch["hazard"].to(self.device)
            taxonomy_idxs = batch.get("taxonomy")
            if taxonomy_idxs is not None:
                taxonomy_idxs = taxonomy_idxs.to(self.device)

            with torch.no_grad():
                t_out = self.teacher(static, seq, mask, taxonomy_idxs)

            s_out = self.student(static, seq, mask, taxonomy_idxs)

            t_probs = F.softmax(
                torch.stack([t_out.hazard_logits[:, h] for h in range(3)], dim=-1) / self.temp, dim=-1
            )
            s_logits = torch.stack([s_out.hazard_logits[:, h] for h in range(3)], dim=-1) / self.temp
            distill = F.kl_div(F.log_softmax(s_logits, dim=-1), t_probs, reduction="batchmean")

            valid = targets >= 0
            hard = F.binary_cross_entropy_with_logits(
                s_out.hazard_logits[valid], targets[valid].clamp(min=0), reduction="mean"
            )

            loss = self.alpha * distill + (1 - self.alpha) * hard

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / max(len(loader), 1)


def main():
    cfg = load_config("config/default.yaml")
    device = get_device()
    feature_dir = Path("data/features")

    seq_df = pl.read_csv(feature_dir / "sequences.tsv", separator="\t")
    seq_means, seq_stds = load_normalizers(feature_dir / "normalizers.npz")
    static_means, static_stds = load_normalizers(feature_dir / "static_normalizers.npz")
    taxonomy_vocab = load_taxonomy_vocab(feature_dir / "taxonomy_vocab.json")
    use_taxonomy = bool(taxonomy_vocab)

    train_df = seq_df.filter((pl.col("split") == "train") & (pl.col("observed") == 1.0))
    max_seq_len = cfg.model.max_seq_len

    train_ds = SequenceDataset(
        train_df, train_df["backbone_id"].unique().to_list(), max_seq_len,
        normalizer=(seq_means, seq_stds), static_normalizer=(static_means, static_stds),
        use_taxonomy=use_taxonomy,
    )
    sampler = SequenceBatchSampler(len(train_ds.items), batch_size=cfg.training.batch_size)
    collate = lambda b: sequence_collate(b, max_seq_len)
    loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate)

    n_static = train_ds.items[0]["static"].size(-1)
    n_snapshot = train_ds.items[0]["seq"].size(-1)

    teacher = create_model(n_static, n_snapshot, cfg.model, taxonomy_vocab if use_taxonomy else None)
    teacher_path = "artifacts/BS_20260516_092852/best_model.pt"
    teacher.load_state_dict(torch.load(teacher_path, map_location=device, weights_only=True))
    logger.info("Loaded teacher from %s", teacher_path)

    student = BioSpreadModel(
        n_static=n_static, n_snapshot=n_snapshot,
        taxonomy_vocab_sizes=None,
        static_dim=cfg.model.static_dim // 2,
        temporal_dim=cfg.model.temporal_dim // 2,
        hidden_dim=cfg.model.gru_hidden // 2,
        num_layers=cfg.model.gru_layers,
        n_hazard=cfg.model.n_hazard_steps,
        max_seq_len=cfg.model.max_seq_len,
        dropout=cfg.model.dropout * 1.5,
        use_enhanced_cold_start=False,
    )

    trainer = DistillationTrainer(teacher, student, device, lr=cfg.training.lr * 0.5)
    for epoch in range(15):
        loss = trainer.train_epoch(loader)
        logger.info("Epoch %2d | Distill Loss: %.4f", epoch + 1, loss)

    out_dir = Path("artifacts/distill")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), out_dir / "student_model.pt")
    logger.info("Student saved to %s", out_dir / "student_model.pt")


if __name__ == "__main__":
    main()
