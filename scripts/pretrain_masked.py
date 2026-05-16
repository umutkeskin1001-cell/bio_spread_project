from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bio_spread.data.dataset import (
    SequenceBatchSampler, SequenceDataset, load_normalizers, sequence_collate,
)
from bio_spread.data.loader import get_device
from bio_spread.data.snapshot import load_taxonomy_vocab
from bio_spread.models import create_model
from bio_spread.utils.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pretrain_masked")


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

    model = create_model(n_static, n_snapshot, cfg.model, taxonomy_vocab if use_taxonomy else None)
    model.enable_pretrain(n_features=n_snapshot, hidden_dim=cfg.model.gru_hidden)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
    mask_ratio = 0.25

    model.train()
    for epoch in range(20):
        total_loss = 0.0
        for batch in loader:
            static = batch["static"].to(device)
            seq = batch["seq"].to(device)
            mask = batch["mask"].to(device)
            taxonomy_idxs = batch.get("taxonomy")
            if taxonomy_idxs is not None:
                taxonomy_idxs = taxonomy_idxs.to(device)

            B, L, F = seq.shape
            flat_mask = mask.bool().flatten()
            n_flat = flat_mask.sum().item()
            if n_flat < 2:
                continue

            n_mask = max(1, int(n_flat * mask_ratio))
            mask_indices = flat_mask.nonzero().squeeze(-1)
            perm = torch.randperm(n_flat, device=seq.device)
            masked_flat = mask_indices[perm[:n_mask]]

            seq_masked = seq.clone()
            seq_masked_flat = seq_masked.reshape(-1, F)
            seq_masked_flat[masked_flat] = 0.0

            out = model(static, seq_masked, mask, taxonomy_idxs)
            h_all, _ = model.temporal_encoder(seq_masked, mask)

            pred = model.pretrain_head(h_all.reshape(-1, h_all.size(-1)))
            target = seq.reshape(-1, F)

            recon_loss = F.mse_loss(pred[masked_flat], target[masked_flat])
            unchanged = flat_mask.nonzero().squeeze(-1)
            unchanged = unchanged[~torch.isin(unchanged, masked_flat)]
            if unchanged.numel() > 0:
                recon_loss += 0.1 * F.mse_loss(pred[unchanged], target[unchanged])

            optimizer.zero_grad()
            recon_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += recon_loss.item()

        scheduler.step()
        avg = total_loss / max(len(loader), 1)
        logger.info("Epoch %2d | Recon Loss: %.6f", epoch + 1, avg)

    out_dir = Path("artifacts/pretrain")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "pretrained_model.pt")
    logger.info("Pre-trained model saved to %s", out_dir / "pretrained_model.pt")


if __name__ == "__main__":
    main()
