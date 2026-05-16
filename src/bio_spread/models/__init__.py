from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bio_spread.config.schema import ModelConfig
from bio_spread.models.sovereign import BioSpreadModel

_TAXONOMY_KEYS: list[str] = [
    "TAXONOMY_phylum", "TAXONOMY_class", "TAXONOMY_order", "TAXONOMY_family", "genus",
]


def _get_vocab_size(vocab: Mapping[str, Any], key: str) -> int:
    values = vocab.get(key)
    if values is None:
        return 2
    n = len(values)
    return max(n + 1, 2)


def create_model(
    n_static: int,
    n_snapshot: int,
    model_cfg: ModelConfig,
    taxonomy_vocab: Mapping[str, Any] | None = None,
    categorical_vocabs: Mapping[str, Any] | None = None,
) -> BioSpreadModel:
    tax_vocab_sizes: list[int] | None = None
    if taxonomy_vocab:
        tax_vocab_sizes = [_get_vocab_size(taxonomy_vocab, k) for k in _TAXONOMY_KEYS]

    cat_vocab_sizes: dict[str, int] | None = None
    if categorical_vocabs:
        cat_vocab_sizes = {}
        for col, vocab in categorical_vocabs.items():
            cat_vocab_sizes[col] = len(vocab) + 1

    return BioSpreadModel(
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
        use_cross_attention=model_cfg.use_cross_attention,
        categorical_vocab_sizes=cat_vocab_sizes,
        categorical_embed_dim=model_cfg.categorical_embed_dim,
        # Sovereign-X Ultra flags
        use_mamba=model_cfg.use_mamba,
        use_hyperbolic=model_cfg.use_hyperbolic,
        use_evidential=model_cfg.use_evidential,
        use_retrieval=model_cfg.use_retrieval,
        mamba_d_state=model_cfg.mamba_d_state,
        mamba_n_layers=model_cfg.mamba_n_layers,
        conv_kernel=model_cfg.conv_kernel,
        tax_dim_per_level=model_cfg.tax_dim_per_level,
        hyperbolic_curvature=model_cfg.hyperbolic_curvature,
        prototype_dim=model_cfg.prototypes,
        prototype_k=model_cfg.prototype_k,
        ema_alpha=model_cfg.ema_alpha,
        edl_lambda_kl=model_cfg.edl_lambda_kl,
        edl_target_smoothing=model_cfg.edl_target_smoothing,
        fit_heads=model_cfg.fit_heads,
        use_research=model_cfg.use_research,
    )


__all__ = ["create_model", "BioSpreadModel"]
