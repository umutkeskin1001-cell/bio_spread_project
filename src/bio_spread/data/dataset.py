from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, Sampler

from bio_spread.constants import ALL_SNAPSHOT_COLS, CATEGORICAL_COLS, COUNT_COL, HAZARD_COLS, HEAVY_TAILED_FEATURES, SNAPSHOT_FEATURE_COLS, SNAPSHOT_NAN_COLS, STATIC_COLS, TAXONOMY_COLS

logger = logging.getLogger(__name__)

UNKNOWN_TAXONOMY_IDX = 1


def fit_normalizers(train_df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    n_base = len(SNAPSHOT_FEATURE_COLS)
    n_nan = len(SNAPSHOT_NAN_COLS)
    n_total = n_base + n_nan

    means = np.zeros(n_total)
    stds = np.ones(n_total)

    avail = [c for c in SNAPSHOT_FEATURE_COLS if c in train_df.columns]
    if avail:
        block = train_df.select(avail).to_numpy()
        col_map = {col: i for i, col in enumerate(avail)}
        for i, col in enumerate(SNAPSHOT_FEATURE_COLS):
            if col not in col_map:
                continue
            vals = block[:, col_map[col]]
            if col in HEAVY_TAILED_FEATURES:
                vals = np.log1p(np.maximum(vals, 0))
            means[i] = np.nanmean(vals)
            stds[i] = np.nanstd(vals).clip(min=1e-8)

    return means, stds


def fit_static_normalizers(train_df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    avail = [c for c in STATIC_COLS if c in train_df.columns]
    missing = [c for c in STATIC_COLS if c not in train_df.columns]
    if missing:
        logger.warning("Missing STATIC_COLS in training data: %s. Filling with zeros.", missing)
    if not avail:
        return np.zeros(len(STATIC_COLS)), np.ones(len(STATIC_COLS))
    first = train_df.unique(subset=["backbone_id"])
    block = first.select(avail).to_numpy()
    arr = np.zeros((block.shape[0], len(STATIC_COLS)))
    col_map = {col: i for i, col in enumerate(avail)}
    for j, col in enumerate(STATIC_COLS):
        if col in col_map:
            arr[:, j] = block[:, col_map[col]]
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0).clip(min=1e-8)


def save_normalizers(path: Path, means: np.ndarray, stds: np.ndarray):
    np.savez_compressed(path, means=means, stds=stds)


def load_normalizers(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["means"], data["stds"]


class SequenceDataset(Dataset):

    def __init__(
        self,
        sequences_df: pl.DataFrame,
        backbone_ids: List[str],
        max_seq_len: int = 50,
        normalizer: Optional[tuple[np.ndarray, np.ndarray]] = None,
        static_normalizer: Optional[tuple[np.ndarray, np.ndarray]] = None,
        use_taxonomy: bool = True,
        use_categorical: bool = True,
    ):
        self.max_seq_len = max_seq_len
        self.use_taxonomy = use_taxonomy and all(c in sequences_df.columns for c in TAXONOMY_COLS)
        cat_cols_present = [c for c in CATEGORICAL_COLS if f"cat_{c}" in sequences_df.columns]
        self.use_categorical = use_categorical and len(cat_cols_present) > 0
        self.cat_cols = cat_cols_present
        self.means, self.stds = normalizer or (
            np.zeros(len(ALL_SNAPSHOT_COLS)),
            np.ones(len(ALL_SNAPSHOT_COLS)),
        )
        if len(self.means) < len(ALL_SNAPSHOT_COLS):
            n_pad = len(ALL_SNAPSHOT_COLS) - len(self.means)
            self.means = np.concatenate([self.means, np.zeros(n_pad)])
            self.stds = np.concatenate([self.stds, np.ones(n_pad)])
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
        n_features = len(ALL_SNAPSHOT_COLS)
        n_base = len(SNAPSHOT_FEATURE_COLS)
        n_static = len(STATIC_COLS)

        snap_cols = [c for c in SNAPSHOT_FEATURE_COLS if c in df.columns]
        static_cols = [c for c in STATIC_COLS if c in df.columns]
        hazard_cols = [c for c in HAZARD_COLS if c in df.columns]

        for bid in backbone_ids:
            g = grouped.get(bid)
            if g is None or len(g) == 0:
                continue
            seq_len = min(len(g), self.max_seq_len)

            feat = np.zeros((seq_len, n_features), dtype=np.float32)
            if snap_cols:
                raw_vals = g.select(snap_cols).to_numpy()[-seq_len:]
                col_map = {col: i for i, col in enumerate(snap_cols)}
                for j, col in enumerate(SNAPSHOT_FEATURE_COLS):
                    if col not in col_map:
                        continue
                    col_vals = raw_vals[:, col_map[col]]
                    nan_mask = np.isnan(col_vals)
                    if col in HEAVY_TAILED_FEATURES:
                        col_vals = np.log1p(np.maximum(col_vals, 0))
                    col_vals = np.where(nan_mask, self.means[j], col_vals)
                    feat[:, j] = col_vals
                    feat[:, n_base + j] = nan_mask.astype(np.float32)
            feat = (feat - self.means) / self.stds

            static_arr = np.zeros(n_static, dtype=np.float32)
            if static_cols:
                raw_static = g.select(static_cols).to_numpy()[0]
                col_map = {col: i for i, col in enumerate(static_cols)}
                for j, col in enumerate(STATIC_COLS):
                    if col not in col_map:
                        continue
                    v = raw_static[col_map[col]]
                    static_arr[j] = float(v) if not math.isnan(float(v)) else 0.0
            static_arr = (static_arr - self.static_means) / self.static_stds

            taxonomy_idxs = None
            if self.use_taxonomy:
                idxs = np.full(5, UNKNOWN_TAXONOMY_IDX, dtype=np.int64)
                if all(c in g.columns for c in TAXONOMY_COLS):
                    raw_tax = g.select(TAXONOMY_COLS).to_numpy()[0]
                    for j, col in enumerate(TAXONOMY_COLS):
                        v = raw_tax[j]
                        idxs[j] = int(v) if v is not None else UNKNOWN_TAXONOMY_IDX
                taxonomy_idxs = idxs

            hazard = np.full((seq_len, 3), -1.0, dtype=np.float32)
            if hazard_cols:
                raw_hazard = g.select(hazard_cols).to_numpy()[-seq_len:]
                col_map = {col: i for i, col in enumerate(hazard_cols)}
                for j, col in enumerate(HAZARD_COLS):
                    if col not in col_map:
                        continue
                    hazard[:, j] = np.nan_to_num(raw_hazard[:, col_map[col]], nan=-1.0)

            count_val = 0.0
            if COUNT_COL in g.columns:
                last_count = g[COUNT_COL].to_numpy()[-1]
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

            if self.use_categorical:
                cat_data = {}
                for col in self.cat_cols:
                    col_key = f"cat_{col}"
                    if col_key in g.columns:
                        val = g[col_key].to_numpy()[0]
                        if val is not None and str(val).strip():
                            try:
                                indices = json.loads(str(val))
                                cat_data[col] = torch.tensor(indices, dtype=torch.long)
                            except (json.JSONDecodeError, ValueError):
                                cat_data[col] = torch.tensor([0], dtype=torch.long)
                        else:
                            cat_data[col] = torch.tensor([0], dtype=torch.long)
                    else:
                        cat_data[col] = torch.tensor([0], dtype=torch.long)
                item["cat_inputs"] = cat_data

            self.items.append(item)

        logger.info(
            "Built SequenceDataset with %d backbones (taxonomy=%s, categorical=%s)",
            len(self.items), self.use_taxonomy, self.use_categorical,
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]


def sequence_collate(batch: List[Dict], max_seq_len: int) -> Dict[str, torch.Tensor]:
    B = len(batch)
    n_features = batch[0]["seq"].size(-1)
    n_static = batch[0]["static"].size(-1)
    has_taxonomy = "taxonomy" in batch[0]
    has_categorical = "cat_inputs" in batch[0]

    seqs = torch.zeros(B, max_seq_len, n_features)
    hazards = torch.full((B, max_seq_len, 3), -1.0)
    masks = torch.zeros(B, max_seq_len)
    lengths = torch.zeros(B, dtype=torch.long)
    static = torch.zeros(B, n_static)
    counts = torch.full((B,), -1.0)
    taxonomy_idxs = torch.ones(B, 5, dtype=torch.long) if has_taxonomy else None
    bids = []

    cat_inputs: dict[str, tuple[list[torch.Tensor], list[int]]] = {}
    if has_categorical:
        sample_cats = batch[0]["cat_inputs"]
        for col in sample_cats:
            cat_inputs[col] = ([], [])

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
        if has_categorical:
            for col, (indices_list, offsets_list) in cat_inputs.items():
                indices = item["cat_inputs"].get(col, torch.tensor([0], dtype=torch.long))
                offsets_list.append(len(indices_list))
                indices_list.append(indices)
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
    if has_categorical:
        cat_tensors = {}
        for col, (indices_list, offsets_list) in cat_inputs.items():
            if indices_list:
                cat_tensors[col] = torch.cat(indices_list)
                cat_tensors[f"{col}_offsets"] = torch.tensor(offsets_list, dtype=torch.long)
            else:
                cat_tensors[col] = torch.zeros(1, dtype=torch.long)
                cat_tensors[f"{col}_offsets"] = torch.zeros(B, dtype=torch.long)
        result["cat_inputs"] = cat_tensors
    return result


class SequenceBatchSampler(Sampler):

    def __init__(self, n_samples: int, batch_size: int, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = n_samples

    def __iter__(self):
        indices = list(range(self.n_samples))
        if self.shuffle:
            np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in range(0, self.n_samples, self.batch_size)]
        if self.shuffle:
            np.random.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        if self.n_samples == 0:
            return 0
        return max(1, (self.n_samples + self.batch_size - 1) // self.batch_size)
