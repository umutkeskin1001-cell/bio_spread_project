from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import polars as pl
import torch
from torch.utils.data import DataLoader

from bio_spread.config.schema import Config
from bio_spread.data.dataset import SequenceDataset, load_normalizers, sequence_collate
from bio_spread.data.snapshot import load_taxonomy_vocab


def load_training_data(
    cfg: Config,
    feature_dir: str,
    filter_observed: bool = True,
) -> Dict[str, Any]:
    feature_dir = Path(feature_dir)

    seq_df = pl.read_csv(feature_dir / "sequences.tsv", separator="\t")
    if filter_observed:
        seq_df = seq_df.filter(pl.col("observed") == 1.0)

    with open(feature_dir / "split.json") as f:
        import json
        split = json.load(f)

    tax_vocab = load_taxonomy_vocab(feature_dir / "taxonomy_vocab.json")
    use_taxonomy = bool(tax_vocab)

    seq_means, seq_stds = load_normalizers(feature_dir / "normalizers.npz")
    static_means, static_stds = load_normalizers(feature_dir / "static_normalizers.npz")

    train_df = seq_df.filter(pl.col("backbone_id").is_in(split["train"]))

    result = {
        "sequences_df": seq_df,
        "split": split,
        "train_df": train_df,
        "seq_means": seq_means,
        "seq_stds": seq_stds,
        "static_means": static_means,
        "static_stds": static_stds,
        "tax_vocab": tax_vocab,
        "use_taxonomy": use_taxonomy,
    }

    return result


def make_sequence_dataset(
    seq_df: pl.DataFrame,
    backbone_ids: list[str],
    cfg: Config,
    seq_means,
    seq_stds,
    static_means,
    static_stds,
    use_taxonomy: bool,
) -> SequenceDataset:
    return SequenceDataset(
        seq_df,
        backbone_ids,
        cfg.model.max_seq_len,
        normalizer=(seq_means, seq_stds),
        static_normalizer=(static_means, static_stds),
        use_taxonomy=use_taxonomy,
    )


def make_data_loader(
    dataset: SequenceDataset,
    batch_size: int,
    max_seq_len: int,
    shuffle: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: sequence_collate(b, max_seq_len),
    )


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
