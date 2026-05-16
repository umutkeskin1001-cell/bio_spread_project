from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from bio_spread.models.components import (
    MLP,
    CategoricalEncoder,
    ColdStartEncoder,
    CountHead,
    EvidentialHazardHead,
    FiTBlock,
    GatedResidualMLP,
    HybridTemporalEncoder,
    HyperbolicFusionGate,
    PoincareTaxonomyEncoder,
    PretrainHead,
    TaxonomyEncoder,
    TemporalPriorPredictor,
    UncertaintyProtoRetriever,
    FiLM,
    TemporalProxyGenerator,
)


@dataclass
class ModelOutput:
    hazard_logits: torch.Tensor
    hazard_logits_all: torch.Tensor
    count_logits: torch.Tensor
    cold_logits: torch.Tensor
    fused: torch.Tensor
    gate_weights: torch.Tensor
    mask: torch.Tensor
    # Sovereign-X Ultra fields
    alpha_pos: torch.Tensor | None = None
    epistemic_var: torch.Tensor | None = None
    routing_weight: torch.Tensor | None = None
    h_static: torch.Tensor | None = None
    h_temporal: torch.Tensor | None = None
    h_cold: torch.Tensor | None = None
    cold_input: torch.Tensor | None = None
    cold_prior: torch.Tensor | None = None
    # Phase 1 new fields
    proxy_temporal: torch.Tensor | None = None
    routing_w: torch.Tensor | None = None


class StaticEncoder(nn.Module):
    def __init__(self, input_dim: int, static_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.encoder = GatedResidualMLP(
            [input_dim, static_dim * 2, static_dim * 2, static_dim],
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class FusionGate(nn.Module):
    def __init__(self, static_dim: int, temporal_dim: int, use_cross_attention: bool = False):
        super().__init__()
        self.use_cross_attention = use_cross_attention
        if static_dim != temporal_dim:
            self.temporal_proj = nn.Linear(temporal_dim, static_dim)
        else:
            self.temporal_proj = nn.Identity()

        gate_input_dim = static_dim + static_dim
        if use_cross_attention:
            gate_input_dim += static_dim
            self.cross_attn_q = nn.Linear(static_dim, static_dim)
            self.cross_attn_k = nn.Linear(temporal_dim, static_dim)
            self.cross_attn_v = nn.Linear(temporal_dim, static_dim)

        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        static: torch.Tensor,
        temporal: torch.Tensor,
        h_all_proj: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        temporal = self.temporal_proj(temporal)
        gate_input = [static, temporal]

        if self.use_cross_attention and h_all_proj is not None and mask is not None:
            q = self.cross_attn_q(static).unsqueeze(1)
            k = self.cross_attn_k(h_all_proj)
            v = self.cross_attn_v(h_all_proj)
            attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(static.size(-1))
            attn = attn.masked_fill(~mask.bool().unsqueeze(1), -1e9)
            attn = F.softmax(attn, dim=-1)
            context = torch.matmul(attn, v).squeeze(1)
            gate_input.append(context)

        weights = self.gate(torch.cat(gate_input, dim=-1))
        fused = weights[:, 0:1] * static + weights[:, 1:2] * temporal
        return fused, weights


class BioSpreadModel(nn.Module):
    def __init__(
        self,
        n_static: int,
        n_snapshot: int,
        taxonomy_vocab_sizes: list[int] | None = None,
        taxonomy_embed_dim: int = 12,
        static_dim: int = 144,
        temporal_dim: int = 144,
        hidden_dim: int = 144,
        num_layers: int = 2,
        n_hazard: int = 3,
        max_seq_len: int = 50,
        dropout: float = 0.15,
        use_cross_attention: bool = False,
        categorical_vocab_sizes: dict[str, int] | None = None,
        categorical_embed_dim: int = 16,
        # Sovereign-X Ultra flags
        use_mamba: bool = False,
        use_hyperbolic: bool = False,
        use_evidential: bool = False,
        use_retrieval: bool = False,
        mamba_d_state: int = 16,
        mamba_n_layers: int = 4,
        conv_kernel: int = 3,
        tax_dim_per_level: int = 18,
        hyperbolic_curvature: float = -1.0,
        prototype_dim: int = 512,
        prototype_k: int = 8,
        ema_alpha: float = 0.992,
        edl_lambda_kl: float = 0.1,
        edl_target_smoothing: float = 0.05,
        fit_heads: int = 4,
        use_research: bool = True,
    ):
        super().__init__()
        self.use_research = use_research
        self.use_taxonomy = taxonomy_vocab_sizes is not None and len(taxonomy_vocab_sizes) > 0
        self.use_cross_attention = use_cross_attention
        self.use_mamba = use_mamba
        self.use_hyperbolic = use_hyperbolic
        self.use_evidential = use_evidential
        self.use_retrieval = use_retrieval if use_research else False
        self.n_hazard = n_hazard
        self.static_dim = static_dim
        self.temporal_dim = temporal_dim
        self.max_seq_len = max_seq_len

        self.use_categorical = categorical_vocab_sizes is not None and len(categorical_vocab_sizes) > 0
        if self.use_categorical:
            self.categorical_encoder = CategoricalEncoder(
                categorical_vocab_sizes, categorical_embed_dim, dropout
            )
            cat_embed_dim = self.categorical_encoder.output_dim
        else:
            cat_embed_dim = 0

        # Taxonomy encoder (Euclidean or Poincaré)
        if self.use_taxonomy:
            if use_hyperbolic:
                self.taxonomy_encoder = PoincareTaxonomyEncoder(
                    taxonomy_vocab_sizes, tax_dim_per_level, dropout, hyperbolic_curvature
                )
            else:
                self.taxonomy_encoder = TaxonomyEncoder(taxonomy_vocab_sizes, taxonomy_embed_dim, dropout)
            tax_embed_total = self.taxonomy_encoder.output_dim
            static_input_dim = n_static + tax_embed_total
        else:
            tax_embed_total = 0
            static_input_dim = n_static

        # Static encoder: GatedResidualMLP or FiT
        if use_hyperbolic:
            self.static_encoder = FiTBlock(static_input_dim, static_dim, fit_heads, dropout)
        else:
            self.static_encoder = StaticEncoder(static_input_dim, static_dim, dropout)

        # Temporal encoder (GRU or Mamba-2 + CausalConv)
        self.temporal_encoder = HybridTemporalEncoder(
            n_snapshot, hidden_dim, num_layers, max_seq_len, dropout,
            use_mamba, mamba_d_state, mamba_n_layers, conv_kernel,
        )
        self.temporal_proj = nn.Linear(hidden_dim, temporal_dim)

        # Fusion gate (Euclidean or Hyperbolic)
        if use_hyperbolic:
            self.gate = HyperbolicFusionGate(static_dim, temporal_dim, hyperbolic_curvature, dropout)
        else:
            self.gate = FusionGate(static_dim, temporal_dim, use_cross_attention)

        fused_dim = static_dim

        # Hazard head (normal or evidential)
        if use_evidential:
            self.hazard_head = EvidentialHazardHead(fused_dim, n_hazard, edl_lambda_kl, edl_target_smoothing)
            self.count_head = CountHead(fused_dim)
        else:
            self.hazard_proj = MLP([fused_dim, fused_dim // 2, fused_dim // 2], dropout=dropout)
            self.hazard_head = nn.Linear(fused_dim // 2, n_hazard)
            self.count_head = CountHead(fused_dim // 2)

        # Per-timestep head
        self.timestep_head = nn.Sequential(
            nn.Linear(static_dim + temporal_dim, static_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(static_dim // 2, n_hazard),
        )

        # --- Phase 1: Cold-Start Revolution ---
        # Temporal Proxy Generator: Static + Taxonomy -> Proxy Temporal
        self.proxy_generator = TemporalProxyGenerator(static_dim, tax_embed_total, temporal_dim, dropout)
        
        # Cold-Start Head: GatedResidualMLP + FiLM conditioning on taxonomy
        # If no taxonomy, tax_embed_total is 0, FiLM acts on empty/identity
        self.cold_film = FiLM(max(tax_embed_total, 1), static_dim)
        self.cold_head = GatedResidualMLP(
            [static_dim + temporal_dim, static_dim, n_hazard],
            dropout=dropout
        )

        # Uncertainty-guided prototypical retrieval (Legacy/Research)
        if self.use_retrieval:
            cold_query_dim = static_dim * 2 + tax_embed_total + cat_embed_dim
            self.retriever = UncertaintyProtoRetriever(
                cold_query_dim, n_hazard, prototype_k, ema_alpha, prototype_dim, static_dim,
            )
            self.routing_tau = nn.Parameter(torch.ones(1) * 2.0)
            self.routing_thr = nn.Parameter(torch.tensor(0.5))
            self.routing_beta = nn.Parameter(torch.ones(1) * 3.0)

        self.pretrain_head: PretrainHead | None = None

    def enable_pretrain(self, n_features: int, hidden_dim: int = 192):
        self.pretrain_head = PretrainHead(hidden_dim, n_features)

    def forward(
        self,
        static: torch.Tensor,
        snapshots: torch.Tensor,
        mask: torch.Tensor,
        taxonomy_idxs: torch.Tensor | None = None,
        temporal_mask: torch.Tensor | None = None,
        cat_inputs: dict[str, torch.Tensor] | None = None,
    ) -> ModelOutput:
        B, L = snapshots.shape[:2]

        # --- Taxonomy encoding ---
        tax_emb = None
        if self.use_taxonomy and taxonomy_idxs is not None:
            tax_emb, _ = self.taxonomy_encoder(taxonomy_idxs)
            static_input = torch.cat([static, tax_emb], dim=-1)
        else:
            static_input = static
        z_static = self.static_encoder(static_input)

        # --- Temporal encoding ---
        h_all, h_pooled = self.temporal_encoder(snapshots, mask, temporal_mask)

        # --- Per-timestep supervision ---
        z_static_exp = z_static.unsqueeze(1).expand(-1, L, -1)
        h_all_proj = self.temporal_proj(h_all)
        ts_input = torch.cat([z_static_exp, h_all_proj], dim=-1)
        hazard_logits_all = self.timestep_head(ts_input)

        # --- Fusion ---
        h_pooled_proj = self.temporal_proj(h_pooled)
        fused, gate_weights = self.gate(z_static, h_pooled_proj, h_all_proj, mask)

        # --- Hazard heads ---
        if self.use_evidential:
            expected_prob, alpha_pos, epistemic_var = self.hazard_head(fused)
            hazard_logits = torch.logit(expected_prob.clamp(1e-7, 1 - 1e-7))
            count_logits = self.count_head(fused)
        else:
            h = self.hazard_proj(fused)
            hazard_logits = self.hazard_head(h)
            count_logits = self.count_head(h)
            alpha_pos = None
            epistemic_var = None

        # --- Phase 1: Cold-Start Path ---
        proxy_temporal = self.proxy_generator(z_static, tax_emb if tax_emb is not None else torch.zeros(B, 0, device=z_static.device))
        
        # FiLM conditioning: cold path adapts to taxonomy
        z_static_conditioned = self.cold_film(z_static, tax_emb if tax_emb is not None else torch.ones(B, 1, device=z_static.device))
        
        cold_combined = torch.cat([z_static_conditioned, proxy_temporal], dim=-1)
        cold_logits = self.cold_head(cold_combined)

        # --- Deterministic Confidence-Gated Routing ---
        # w = sigmoid(k*(thr - seq_len)) * (1 - entropy(temporal_logits))
        seq_lens = mask.sum(dim=1)
        # Use a soft threshold around 3 snapshots for "cold"
        len_gate = torch.sigmoid(2.0 * (3.5 - seq_lens)) 
        
        # Calculate entropy of temporal prediction if possible, else 1.0 (max uncertainty)
        if self.use_evidential and epistemic_var is not None:
            # Normalized epistemic uncertainty as entropy proxy
            entropy_proxy = epistemic_var.mean(dim=-1).clamp(0, 1)
        else:
            # Fallback to simple seq_len based gating if not evidential
            entropy_proxy = torch.ones_like(len_gate)
            
        routing_w = (len_gate * entropy_proxy).unsqueeze(-1)
        # Clamp routing to avoid complete cut-off during training
        routing_w = routing_w.clamp(0.01, 0.99)
        
        final_hazard_logits = (1 - routing_w) * hazard_logits + routing_w * cold_logits
        
        # Research/Legacy Retrieval Fallback
        h_cold = None
        if self.use_research and self.use_retrieval:
            # (Keeping old retrieval logic for research comparison if needed)
            ret_input = torch.cat([z_static, proxy_temporal, tax_emb], dim=-1) if tax_emb is not None else torch.cat([z_static, proxy_temporal], dim=-1)
            ret_logits = self.retriever(ret_input)
            final_hazard_logits = 0.8 * final_hazard_logits + 0.2 * ret_logits
            h_cold = ret_logits

        return ModelOutput(
            hazard_logits=final_hazard_logits,
            hazard_logits_all=hazard_logits_all,
            count_logits=count_logits,
            cold_logits=cold_logits,
            fused=fused,
            gate_weights=gate_weights,
            mask=mask,
            alpha_pos=alpha_pos,
            epistemic_var=epistemic_var,
            routing_weight=routing_w.squeeze(-1),
            h_static=z_static,
            h_temporal=h_pooled_proj,
            h_cold=h_cold,
            proxy_temporal=proxy_temporal,
        )

    def get_embedding(
        self,
        static: torch.Tensor,
        snapshots: torch.Tensor,
        mask: torch.Tensor,
        taxonomy_idxs: torch.Tensor | None = None,
        temporal_mask: torch.Tensor | None = None,
        cat_inputs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.use_taxonomy and taxonomy_idxs is not None:
            tax_emb, _ = self.taxonomy_encoder(taxonomy_idxs)
            static_input = torch.cat([static, tax_emb], dim=-1)
        else:
            static_input = static
        z_static = self.static_encoder(static_input)
        _, h_pooled = self.temporal_encoder(snapshots, mask, temporal_mask)
        h_pooled_proj = self.temporal_proj(h_pooled)
        fused, _ = self.gate(z_static, h_pooled_proj)
        return fused
