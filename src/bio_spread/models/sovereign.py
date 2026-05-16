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
    ):
        super().__init__()
        self.use_taxonomy = taxonomy_vocab_sizes is not None and len(taxonomy_vocab_sizes) > 0
        self.use_cross_attention = use_cross_attention
        self.use_mamba = use_mamba
        self.use_hyperbolic = use_hyperbolic
        self.use_evidential = use_evidential
        self.use_retrieval = use_retrieval
        self.n_hazard = n_hazard
        self.static_dim = static_dim
        self.temporal_dim = temporal_dim

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

        # Cold-start encoder (uses z_static = static encoder output)
        self.cold_prior_predictor = TemporalPriorPredictor(static_dim)
        cold_input_dim = static_dim * 2 + tax_embed_total + cat_embed_dim
        self.cold_start_encoder = ColdStartEncoder(
            input_dim=cold_input_dim,
            static_dim=static_dim,
            n_hazard=n_hazard,
            dropout=dropout,
        )
        self.cold_aux_head = nn.Sequential(
            nn.Linear(static_dim, static_dim // 2),
            nn.LayerNorm(static_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(static_dim // 2, n_hazard),
        )

        # Uncertainty-guided prototypical retrieval (cold path)
        if use_retrieval:
            cold_query_dim = static_dim * 2 + tax_embed_total + cat_embed_dim
            self.retriever = UncertaintyProtoRetriever(
                cold_query_dim, n_hazard, prototype_k, ema_alpha, prototype_dim, static_dim,
            )
            self.routing_tau = nn.Parameter(torch.ones(1) * 2.0)
            self.routing_thr = nn.Parameter(torch.tensor(0.5))
            # v3+ Adaptive Beta for routing control
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

        # --- Cold-start path (z_static + temporal prior + taxonomy) ---
        cold_prior_pred = self.cold_prior_predictor(z_static)
        cold_input_parts = [z_static, cold_prior_pred]
        if tax_emb is not None:
            cold_input_parts.append(tax_emb)
        if self.use_categorical and cat_inputs is not None:
            cat_emb = self.categorical_encoder(cat_inputs)
            cold_input_parts.append(cat_emb)
        
        cold_input = torch.cat(cold_input_parts, dim=-1)
        cold_logits_main = self.cold_start_encoder(cold_input)
        
        # Enhanced Cold Aux Head with z_static interaction
        cold_logits_aux = self.cold_aux_head(z_static)
        cold_logits = 0.6 * cold_logits_main + 0.4 * cold_logits_aux

        # --- Retrieval-augmented cold path (if uncertain) ---
        h_cold = None
        routing_weight = None
        if self.use_retrieval:
            ret_logits = self.retriever(cold_input)
            if self.use_evidential and epistemic_var is not None and alpha_pos is not None:
                # v3 Soft-Gate: Balanced routing between paths
                epistemic_score = epistemic_var.mean(dim=-1)
                # evidence_strength: Higher means temporal path is very sure
                evidence_strength = alpha_pos.sum(dim=-1) / (alpha_pos.sum(dim=-1) + 1.0)
                
                # v3+ Adaptive β-weighted soft routing: Trust cold-path ONLY if epistemic uncertainty is high AND temporal evidence is low
                route_w = torch.sigmoid(self.routing_beta * (epistemic_score - 0.5) * (1.0 - evidence_strength))
                route_w = route_w.clamp(min=0.01, max=0.90) 
                
                final_logits = (1 - route_w).unsqueeze(-1) * hazard_logits + route_w.unsqueeze(-1) * ret_logits
                routing_weight = route_w
                
                if temporal_mask is not None:
                    # For purely cold samples (forced), we still trust cold-path + retrieval more
                    # but we keep a small residual from the prior-projected temporal head
                    cold_conf = (1.0 - torch.sigmoid(epistemic_score * 2.0)).clamp(min=0.1, max=0.8)
                    cold_blended = cold_conf.unsqueeze(-1) * cold_logits + (1 - cold_conf).unsqueeze(-1) * ret_logits
                    
                    final_logits = torch.where(
                        temporal_mask.unsqueeze(-1),
                        cold_blended,
                        final_logits,
                    )
                    routing_weight = torch.where(temporal_mask, 1.0 - cold_conf, route_w)
                
                hazard_logits = final_logits
            h_cold = ret_logits

        return ModelOutput(
            hazard_logits=hazard_logits,
            hazard_logits_all=hazard_logits_all,
            count_logits=count_logits,
            cold_logits=cold_logits,
            fused=fused,
            gate_weights=gate_weights,
            mask=mask,
            alpha_pos=alpha_pos,
            epistemic_var=epistemic_var,
            routing_weight=routing_weight,
            h_static=z_static,
            h_temporal=h_pooled,
            h_cold=h_cold,
            cold_input=cold_input,
            cold_prior=cold_prior_pred,
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
