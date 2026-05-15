"""Sovereign-X model package."""

from __future__ import annotations

from typing import Any

from bio_spread_reborn.config.schema import ModelConfig
from bio_spread_reborn.models.sovereign import SovereignX

# Taxonomy column keys — MUST match snapshot.py's build_taxonomy_vocab output
_TAXONOMY_KEYS = ["TAXONOMY_phylum", "TAXONOMY_class", "TAXONOMY_order", "TAXONOMY_family", "genus"]


def create_model(
    n_static: int,
    n_snapshot: int,
    model_cfg: ModelConfig,
    taxonomy_vocab: dict[str, Any] | None = None,
) -> SovereignX:
    """Build a SovereignX model from config + inferred dimensions.

    Args:
        n_static: Number of numeric static (backbone-level) features.
        n_snapshot: Number of per-snapshot (time-varying) features.
        model_cfg: Model hyperparameters from config.
        taxonomy_vocab: Optional taxonomy vocabulary dict from
            ``build_taxonomy_vocab()``. If provided, model will include
            taxonomy embeddings.

    Returns:
        SovereignX model (on CPU, needs ``.to(device)``).
    """
    tax_vocab_sizes: list[int] | None = None
    if taxonomy_vocab:
        tax_vocab_sizes = [len(taxonomy_vocab.get(k, {})) for k in _TAXONOMY_KEYS]

    return SovereignX(
        n_static=n_static,
        n_snapshot=n_snapshot,
        taxonomy_vocab_sizes=tax_vocab_sizes,
        taxonomy_embed_dim=model_cfg.taxonomy_embed_dim,
        static_dim=model_cfg.static_dim,
        temporal_dim=model_cfg.temporal_dim,
        hidden_dim=model_cfg.gru_hidden,
        num_layers=model_cfg.gru_layers,
        n_hazard=model_cfg.n_hazard_steps,
        max_seq_len=model_cfg.max_seq_len,
        dropout=model_cfg.dropout,
    )


__all__ = ["create_model", "SovereignX"]
