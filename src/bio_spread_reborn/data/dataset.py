"""
Sovereign-X: Backbone-level sequence dataset (with taxonomy support).

Each item = one backbone's full temporal sequence of snapshots.
Model processes sequences, not individual snapshots.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)

from bio_spread_reborn.data.snapshot import (
    SNAPSHOT_FEATURE_COLS,
    STATIC_COLS,
    TAXONOMY_COLS,
)

# Hazard target columns
HAZARD_COLS = ["hazard_1", "hazard_2", "hazard_3"]

# Count target column
COUNT_COL = "n_new_countries"


def fit_normalizers(train_df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Fit feature normalizers on TRAINING data only. Returns (means, stds)."""
    vals = []
    for col in SNAPSHOT_FEATURE_COLS:
        if col in train_df.columns:
            vals.append(train_df[col].to_numpy())
    if not vals:
        return np.zeros(len(SNAPSHOT_FEATURE_COLS)), np.ones(len(SNAPSHOT_FEATURE_COLS))
    arr = np.column_stack(vals)
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0).clip(min=1e-8)


def fit_static_normalizers(train_df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Fit static feature normalizers on TRAINING data only."""
    first = train_df.group_by("backbone_id").agg([pl.col(c).first().alias(c) for c in STATIC_COLS])
    vals = []
    for col in STATIC_COLS:
        vals.append(first[col].to_numpy())
    if not vals:
        return np.zeros(len(STATIC_COLS)), np.ones(len(STATIC_COLS))
    arr = np.column_stack(vals)
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0).clip(min=1e-8)


def save_normalizers(path: Path, means: np.ndarray, stds: np.ndarray):
    np.savez_compressed(path, means=means, stds=stds)


def load_normalizers(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["means"], data["stds"]


class SovereignSequenceDataset(Dataset):
    """Dataset returning one backbone's temporal sequence per item.

    Supports taxonomy indices as additional static features.
    Normalization is fit on train data only (via fit_normalizers).
    """

    def __init__(
        self,
        sequences_df: pl.DataFrame,
        backbone_ids: List[str],
        max_seq_len: int = 50,
        normalizer: Optional[tuple[np.ndarray, np.ndarray]] = None,
        static_normalizer: Optional[tuple[np.ndarray, np.ndarray]] = None,
        use_taxonomy: bool = True,
    ):
        self.max_seq_len = max_seq_len
        self.use_taxonomy = use_taxonomy and all(c in sequences_df.columns for c in TAXONOMY_COLS)
        self.means, self.stds = normalizer or (
            np.zeros(len(SNAPSHOT_FEATURE_COLS)),
            np.ones(len(SNAPSHOT_FEATURE_COLS)),
        )
        self.static_means, self.static_stds = static_normalizer or (
            np.zeros(len(STATIC_COLS)),
            np.ones(len(STATIC_COLS)),
        )
        self._build(sequences_df, backbone_ids)

    def _build(self, df: pl.DataFrame, backbone_ids: List[str]):
        grouped = {}
        for (bid,), g in df.group_by(["backbone_id"], maintain_order=True):
            if bid in backbone_ids:
                grouped[bid] = g.sort("year")
        self.items = []
        n_features = len(SNAPSHOT_FEATURE_COLS)
        n_static = len(STATIC_COLS)

        for bid in backbone_ids:
            g = grouped.get(bid)
            if g is None or len(g) == 0:
                continue
            seq_len = min(len(g), self.max_seq_len)

            # Extract + normalize snapshot features
            feat = np.zeros((seq_len, n_features), dtype=np.float32)
            for j, col in enumerate(SNAPSHOT_FEATURE_COLS):
                if col in g.columns:
                    vals = g[col].to_numpy()[:seq_len]
                    feat[:, j] = np.nan_to_num(vals, nan=0.0)
            feat = (feat - self.means) / self.stds

            # Extract numeric static features (first snapshot per backbone)
            static_arr = np.zeros(n_static, dtype=np.float32)
            for j, col in enumerate(STATIC_COLS):
                if col in g.columns:
                    v = g[col].to_numpy()[0]
                    static_arr[j] = float(v) if v is not None and v == v else 0.0
            static_arr = (static_arr - self.static_means) / self.static_stds

            # Extract taxonomy indices (NOT normalized, stored as int)
            taxonomy_idxs = None
            if self.use_taxonomy:
                idxs = np.zeros(5, dtype=np.int64)
                for j, col in enumerate(TAXONOMY_COLS):
                    if col in g.columns:
                        v = g[col].to_numpy()[0]
                        idxs[j] = int(v) if v is not None else 0
                taxonomy_idxs = idxs

            # Extract hazards
            hazard = np.full((seq_len, 3), -1.0, dtype=np.float32)
            for j, col in enumerate(HAZARD_COLS):
                if col in g.columns:
                    vals = g[col].to_numpy()[:seq_len]
                    hazard[:, j] = np.nan_to_num(vals, nan=-1.0)

            # Extract count target (last snapshot's value)
            count_val = 0.0
            if COUNT_COL in g.columns:
                last_count = g[COUNT_COL].to_numpy()[seq_len - 1] if seq_len > 0 else 0.0
                count_val = float(last_count) if last_count is not None and last_count >= 0 else -1.0

            item = {
                "seq": torch.from_numpy(feat),
                "static": torch.from_numpy(static_arr),
                "hazard": torch.from_numpy(hazard),
                "count": torch.tensor(count_val, dtype=torch.float32),
                "seq_len": seq_len,
                "backbone_id": bid,
            }
            if self.use_taxonomy:
                item["taxonomy"] = torch.from_numpy(taxonomy_idxs).long()

            self.items.append(item)

        logger.info(
            "Built SovereignSequenceDataset with %d backbones (taxonomy=%s)", len(self.items), self.use_taxonomy
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]


def sequence_collate(batch: List[Dict], max_seq_len: int = 50) -> Dict[str, torch.Tensor]:
    """Collate backbone sequences into padded batch. All tensors on CPU."""
    B = len(batch)
    n_features = batch[0]["seq"].size(-1)
    n_static = batch[0]["static"].size(-1)
    has_taxonomy = "taxonomy" in batch[0]

    seqs = torch.zeros(B, max_seq_len, n_features)
    hazards = torch.full((B, max_seq_len, 3), -1.0)
    masks = torch.zeros(B, max_seq_len)
    lengths = torch.zeros(B, dtype=torch.long)
    static = torch.zeros(B, n_static)
    counts = torch.full((B,), -1.0)
    taxonomy_idxs = torch.zeros(B, 5, dtype=torch.long) if has_taxonomy else None
    bids = []

    for i, item in enumerate(batch):
        L = item["seq_len"]
        lengths[i] = L
        masks[i, :L] = 1.0
        seqs[i, :L] = item["seq"][:L]
        item_h = item["hazard"][:L]
        hazards[i, :L] = item_h
        static[i] = item["static"]
        counts[i] = item["count"]
        if has_taxonomy:
            taxonomy_idxs[i] = item["taxonomy"]
        bids.append(item["backbone_id"])

    result = {
        "seq": seqs,
        "static": static,
        "hazard": hazards,
        "count": counts,
        "mask": masks,
        "seq_len": lengths,
        "backbone_ids": bids,
    }
    if has_taxonomy:
        result["taxonomy"] = taxonomy_idxs
    return result


class SequenceBatchSampler(Sampler):
    """Samples backbones ensuring diverse sequence lengths per batch."""

    def __init__(self, seq_lens: List[int], batch_size: int, shuffle: bool = True):
        self.seq_lens = np.array(seq_lens)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(seq_lens)

    def __iter__(self):
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        sorted_idx = indices[np.argsort(self.seq_lens[indices])]
        batches = []
        for i in range(0, len(sorted_idx), self.batch_size):
            batches.append(sorted_idx[i : i + self.batch_size])
        if self.shuffle:
            np.random.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return max(1, (self.n_samples + self.batch_size - 1) // self.batch_size)
