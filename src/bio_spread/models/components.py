from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    """Feature-wise Linear Modulation."""
    def __init__(self, cond_dim: int, output_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, output_dim)
        self.beta = nn.Linear(cond_dim, output_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma(cond)
        beta = self.beta(cond)
        return gamma * x + beta


class TemporalProxyGenerator(nn.Module):
    """Static + Taxonomy -> MLP -> proxy_temporal_emb (128D)."""
    def __init__(self, static_dim: int, tax_dim: int, proxy_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.net = GatedResidualMLP(
            [static_dim + tax_dim, proxy_dim * 2, proxy_dim],
            dropout=dropout
        )

    def forward(self, z_static: torch.Tensor, tax_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_static, tax_emb], dim=-1)
        return self.net(x)


class MLP(nn.Module):
    def __init__(
        self, dims: list[int], dropout: float = 0.1, activation: type[nn.Module] = nn.ReLU
    ):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedResidualMLP(nn.Module):
    def __init__(self, dims: list[int], dropout: float = 0.15):
        super().__init__()
        assert len(dims) >= 2
        self.main = MLP(dims, dropout=dropout)
        in_dim, out_dim = dims[0], dims[-1]
        self.gate = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Sigmoid(),
        )
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate(x)
        main_out = self.main(x)
        skip_out = self.skip(x)
        return gate * main_out + (1 - gate) * skip_out


class ColdStartHead(nn.Module):
    def __init__(self, static_dim: int, n_hazard: int = 3, dropout: float = 0.1):
        super().__init__()
        self.net = MLP([static_dim, static_dim // 2, n_hazard], dropout=dropout)

    def forward(self, z_static: torch.Tensor) -> torch.Tensor:
        return self.net(z_static)


class EnhancedColdStartHead(nn.Module):
    def __init__(self, static_dim: int, tax_embed_dim: int, n_hazard: int = 3, dropout: float = 0.1):
        super().__init__()
        input_dim = static_dim + tax_embed_dim
        self.net = GatedResidualMLP([input_dim, input_dim // 2, n_hazard], dropout=dropout)

    def forward(self, z_static: torch.Tensor, tax_embed: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_static, tax_embed], dim=-1)
        return self.net(x)


class ColdStartEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 168,
        static_dim: int = 128,
        n_hazard: int = 3,
        dropout: float = 0.18, 
    ):
        super().__init__()
        self.static_dim = static_dim
        self.input_proj = nn.Linear(input_dim, 256)
        self.norm1 = nn.LayerNorm(256)
        
        self.static_proj = nn.Linear(static_dim, 256)
        self.aux_proj = nn.Linear(input_dim - static_dim, 256)
        
        self.self_attn = nn.MultiheadAttention(256, num_heads=8, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(256, num_heads=8, dropout=dropout, batch_first=True)
        
        self.ffn = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
        )
        self.norm2 = nn.LayerNorm(256)
        self.norm3 = nn.LayerNorm(256)
        
        self.heads = nn.ModuleList([
            MLP([256, 128, 1], dropout=dropout) for _ in range(n_hazard)
        ])
        self.residual_proj = nn.Linear(input_dim, 256) if input_dim != 256 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        h_init = self.input_proj(x)
        h_init = self.norm1(h_init)
        h_res = self.residual_proj(x)
        
        # Split features for cross-attention interaction
        f_static = self.static_proj(x[:, :self.static_dim]).unsqueeze(1)
        f_aux = self.aux_proj(x[:, self.static_dim:]).unsqueeze(1)
        f_seq = torch.cat([f_static, f_aux], dim=1)
        
        # Self-attention between feature groups
        attn_s, _ = self.self_attn(f_seq, f_seq, f_seq)
        f_seq = self.norm2(f_seq + attn_s)
        
        # Cross-attention: Static queries Aux
        attn_c, _ = self.cross_attn(f_static, f_aux, f_aux)
        h = self.norm3(f_static.squeeze(1) + attn_c.squeeze(1) + self.ffn(f_seq.mean(dim=1)))
        
        # Ensure gradient flow for h_init (input_proj + norm1) and h_res (residual_proj)
        h = h + h_res + 0.01 * h_init.mean(dim=-1, keepdim=True)
        
        logits = [head(h) for head in self.heads]
        return torch.cat(logits, dim=-1)


class PlattScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_a = nn.Parameter(torch.log(torch.tensor(1.0)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1), requires_grad=True)

    @property
    def a(self) -> torch.Tensor:
        return torch.exp(self.log_a)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.a * logits + self.b


class BetaCalibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
        log_odds = torch.log(p / (1 - p))
        return self.a * log_odds + self.b


class AdaptiveLossWeighting(nn.Module):
    def __init__(self, n_losses: int = 6):
        super().__init__()
        self.log_sigmas = nn.Parameter(torch.zeros(n_losses))

    def forward(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        total = 0.0
        for i, (name, loss) in enumerate(losses.items()):
            if name == "total":
                continue
            sigma = torch.exp(self.log_sigmas[i])
            total += loss / (2 * sigma ** 2) + torch.log(sigma)
        return total


class TemporalContrastiveHead(nn.Module):
    def __init__(self, embed_dim: int = 128, proj_dim: int = 64):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(z), dim=-1)


class PretrainHead(nn.Module):
    def __init__(self, hidden_dim: int, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CountHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, max(input_dim // 2, 1)),
            nn.ReLU(),
            nn.Linear(max(input_dim // 2, 1), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _zero_idx1_hook(grad: torch.Tensor) -> torch.Tensor:
    grad = grad.clone()
    grad[1] = 0.0
    return grad


class TaxonomyEncoder(nn.Module):
    def __init__(self, vocab_sizes: list[int], embed_dim: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(v, embed_dim, padding_idx=0) for v in vocab_sizes])
        for emb in self.embeddings:
            if emb.num_embeddings > 1:
                emb.weight.data[1] = torch.zeros(embed_dim)
                emb.weight.register_hook(_zero_idx1_hook)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = len(vocab_sizes) * embed_dim

    def forward(self, idxs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embs = [emb(idxs[..., i]) for i, emb in enumerate(self.embeddings)]
        out = torch.cat(embs, dim=-1)
        zero_mask = (idxs == 0) | (idxs == 1)
        diversity = (~zero_mask).sum(dim=-1).float() / len(self.embeddings)
        return self.dropout(out), diversity


class CategoricalEncoder(nn.Module):
    def __init__(self, vocab_sizes: dict[str, int], embed_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.col_names = sorted(vocab_sizes.keys())
        self.embeddings = nn.ModuleDict()
        total_dim = 0
        for col in self.col_names:
            v = vocab_sizes[col]
            self.embeddings[col] = nn.Embedding(
                num_embeddings=v, embedding_dim=embed_dim, padding_idx=0
            )
            total_dim += embed_dim
        self.output_dim = total_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, cat_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        device = next(iter(self.embeddings.values())).weight.device
        B = None
        embs = []
        for col in self.col_names:
            indices = cat_inputs.get(col)
            offsets = cat_inputs.get(f"{col}_offsets")
            if indices is not None and offsets is not None:
                if B is None:
                    B = offsets.size(0)
                emb_all = self.embeddings[col](indices.to(device))
                emb = self._mean_pool(emb_all, offsets.to(device), B)
                embs.append(emb)
            else:
                if B is None:
                    B = 1
                emb_dim = self.embeddings[col].embedding_dim
                embs.append(torch.zeros(B, emb_dim, device=device))
        out = torch.cat(embs, dim=-1)
        return self.dropout(out)

    def _mean_pool(self, emb_all: torch.Tensor, offsets: torch.Tensor, B: int) -> torch.Tensor:
        offsets_list = offsets.tolist()
        lengths = []
        for i in range(len(offsets_list)):
            start = offsets_list[i]
            end = offsets_list[i + 1] if i + 1 < len(offsets_list) else len(emb_all)
            lengths.append(end - start)
        lengths = torch.tensor(lengths, device=emb_all.device).clamp(min=1)
        expanded = torch.arange(len(emb_all), device=emb_all.device)
        batch_idx = torch.searchsorted(offsets, expanded, right=True) - 1
        batch_idx = batch_idx.clamp(max=B - 1)
        summed = torch.zeros(B, emb_all.size(-1), device=emb_all.device)
        summed.scatter_add_(0, batch_idx.unsqueeze(-1).expand(-1, emb_all.size(-1)), emb_all)
        return summed / lengths.unsqueeze(-1)


@contextmanager
def mcdropout(model: nn.Module) -> Generator[None, None, None]:
    dropout_states = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            dropout_states[name] = module.training
            module.train()
    try:
        yield
    finally:
        for name, module in model.named_modules():
            if isinstance(module, nn.Dropout) and name in dropout_states:
                if dropout_states[name]:
                    module.train()
                else:
                    module.eval()


def predict_with_uncertainty(
    model: nn.Module,
    x_static: torch.Tensor,
    x_seq: torch.Tensor,
    x_mask: torch.Tensor,
    taxonomy_idxs: torch.Tensor | None = None,
    n_samples: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_probs = []
    with torch.no_grad():
        with mcdropout(model):
            for _ in range(n_samples):
                out = model(x_static, x_seq, x_mask, taxonomy_idxs)
                all_probs.append(torch.sigmoid(out.hazard_logits))
    stacked = torch.stack(all_probs, dim=0)
    return stacked.mean(dim=0), stacked.std(dim=0)


def contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    B = z1.size(0)
    sim = z1 @ z2.T / temperature
    labels = torch.arange(B, device=z1.device)
    loss = F.cross_entropy(sim, labels)
    return loss


class PrototypicalColdStart(nn.Module):
    def __init__(self, backbone_embeddings: torch.Tensor, labels: torch.Tensor, n_horizons: int = 3, k: int = 10):
        super().__init__()
        self.register_buffer("embeddings", backbone_embeddings)
        self.register_buffer("labels", labels)
        self.k = k

    @torch.no_grad()
    def predict(self, query_embedding: torch.Tensor) -> torch.Tensor:
        sim = F.normalize(query_embedding, dim=-1) @ F.normalize(self.embeddings, dim=-1).T
        weights, indices = torch.topk(sim, min(self.k, sim.size(-1)), dim=-1)
        weights = F.softmax(weights / 0.1, dim=-1)
        neighbor_labels = self.labels[indices]
        pred = (weights.unsqueeze(-1) * neighbor_labels).sum(dim=1)
        return pred


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation, groups=in_channels)
        self.proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        x_conv = self.conv(x_t)
        x_conv = x_conv[:, :, :x.size(1)]
        out = x_conv.transpose(1, 2)
        return out + self.proj(x)


class Mamba2Block(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = d_model * expand
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state

        self.norm = nn.RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2)
        self.conv = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1)
        self.dt = nn.Linear(d_inner, d_inner)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)))
        self.B_proj = nn.Linear(d_inner, d_state)
        self.C_proj = nn.Linear(d_inner, d_state)
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model)

    def _selective_scan(self, x: torch.Tensor, delta: torch.Tensor,
                        B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        B_batch, L, D = x.shape
        N = B.shape[-1]
        A = -torch.exp(self.A_log)
        h = torch.zeros(B_batch, D, N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            d = delta[:, t].unsqueeze(-1)
            A_bar = torch.exp(d * A.view(1, 1, -1))
            B_bar = d * B[:, t].unsqueeze(1)
            x_t = x[:, t].unsqueeze(-1)
            h = A_bar * h + B_bar * x_t
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1)
            ys.append(y_t)
        return torch.stack(ys, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_conv = x_in.transpose(1, 2)
        x_conv = self.conv(x_conv)[..., :L]
        x_act = F.silu(x_conv.transpose(1, 2))
        delta = F.softplus(self.dt(x_act))
        B_param = self.B_proj(x_act)
        C_param = self.C_proj(x_act)
        y = self._selective_scan(x_act, delta, B_param, C_param)
        y = y + self.D * x_act
        y = y * F.silu(z)
        return self.out_proj(y) + residual


class HybridTemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 192, num_layers: int = 2,
                 max_seq_len: int = 50, dropout: float = 0.15,
                 use_mamba: bool = False, mamba_d_state: int = 16, mamba_n_layers: int = 4,
                 conv_kernel: int = 3):
        super().__init__()
        self.use_mamba = use_mamba
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.null_embed = nn.Parameter(torch.randn(1, hidden_dim) * 0.01)

        if use_mamba:
            self.conv = CausalConv1d(hidden_dim, hidden_dim, conv_kernel)
            self.blocks = nn.ModuleList([
                Mamba2Block(hidden_dim, d_state=mamba_d_state) for _ in range(mamba_n_layers)
            ])
        else:
            self.gru = nn.GRU(
                hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                temporal_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = x.shape
        h = self.input_proj(x)

        if self.use_mamba:
            h = self.conv(h)
            for block in self.blocks:
                h = block(h)
            h_all = h
        else:
            lens = mask.sum(dim=1).long()
            if lens.max() > 0:
                packed = nn.utils.rnn.pack_padded_sequence(h, lens.cpu(), batch_first=True, enforce_sorted=False)
                packed_out, _ = self.gru(packed)
                h_all, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=L)
            else:
                h_all, _ = self.gru(h)

        scores = self.attn(h_all).squeeze(-1)
        scores = scores.masked_fill(~mask.bool(), -1e9)
        weights = F.softmax(scores, dim=-1)
        h_pooled = (h_all * weights.unsqueeze(-1)).sum(dim=1)

        if temporal_mask is not None and temporal_mask.any():
            mask_t = temporal_mask.unsqueeze(1).unsqueeze(-1)
            h_all = torch.where(mask_t, self.null_embed.unsqueeze(0), h_all)
            h_pooled = torch.where(temporal_mask.unsqueeze(-1), self.null_embed, h_pooled)

        return h_all, h_pooled


class PoincareBall:
    @staticmethod
    def _sqrt_c(c: float = -1.0) -> float:
        from math import sqrt
        return sqrt(abs(c))

    @staticmethod
    def expmap0(u: torch.Tensor, c: float = -1.0) -> torch.Tensor:
        sqrt_c = PoincareBall._sqrt_c(c)
        norm_u = u.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        r = torch.tanh(sqrt_c * norm_u) / (sqrt_c + 1e-8)
        r = r.clamp(max=1.0 - 1e-6)
        return r * u / norm_u

    @staticmethod
    def logmap0(p: torch.Tensor, c: float = -1.0) -> torch.Tensor:
        sqrt_c = PoincareBall._sqrt_c(c)
        norm_p = p.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = torch.atanh(sqrt_c * norm_p.clamp(max=0.99)) / (sqrt_c * norm_p + 1e-8)
        return scale * p

    @staticmethod
    def dist(x: torch.Tensor, y: torch.Tensor, c: float = -1.0) -> torch.Tensor:
        sqrt_c = PoincareBall._sqrt_c(c)
        x_norm_sq = x.norm(dim=-1, keepdim=True).pow(2).clamp(max=0.99)
        y_norm_sq = y.norm(dim=-1, keepdim=True).pow(2).clamp(max=0.99)
        num = (1 + 2 * sqrt_c ** 2 * (x * y).sum(dim=-1, keepdim=True) + sqrt_c ** 4 * x_norm_sq * y_norm_sq).clamp(min=1e-8)
        denom = (1 - sqrt_c ** 2 * x_norm_sq) * (1 - sqrt_c ** 2 * y_norm_sq)
        arg = (num / denom.clamp(min=1e-8)).sqrt()
        return (2 / (sqrt_c + 1e-8)) * torch.atanh(sqrt_c * arg.clamp(max=0.99))

    @staticmethod
    def radius_clamp(x: torch.Tensor, max_r: float = 0.99, c: float = -1.0) -> torch.Tensor:
        sqrt_c = PoincareBall._sqrt_c(c)
        norm_x = x.norm(dim=-1, keepdim=True)
        max_norm = max_r / (sqrt_c + 1e-8)
        scale = torch.where(norm_x > max_norm, max_norm / norm_x.clamp(min=1e-8), torch.ones_like(norm_x))
        return x * scale


class PoincareTaxonomyEncoder(nn.Module):
    def __init__(self, vocab_sizes: list[int], embed_dim: int = 16, dropout: float = 0.1,
                 curvature: float = -1.0, n_levels: int = 5):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(v, embed_dim, padding_idx=0) for v in vocab_sizes
        ])
        for emb in self.embeddings:
            if emb.num_embeddings > 1:
                emb.weight.data[1] = torch.zeros(embed_dim)
                emb.weight.register_hook(_zero_idx1_hook)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = len(vocab_sizes) * embed_dim
        self.curvature = curvature
        self.n_levels = n_levels

    def forward(self, idxs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = idxs.size(0)
        all_embs = []
        for i, emb in enumerate(self.embeddings):
            e = emb(idxs[..., i])
            all_embs.append(e)
        stacked = torch.stack(all_embs, dim=1)
        stacked = PoincareBall.radius_clamp(stacked, max_r=0.95, c=self.curvature)
        out = stacked.reshape(B, -1)
        zero_mask = (idxs == 0) | (idxs == 1)
        diversity = (~zero_mask).sum(dim=-1).float() / self.n_levels
        return self.dropout(out), diversity


class FiTBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim), nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.input_proj(x).unsqueeze(1)
        attn_out, _ = self.cross_attn(x_proj, x_proj, x_proj)
        x_proj = self.norm1(x_proj + attn_out)
        ffn_out = self.ffn(x_proj)
        x_proj = self.norm2(x_proj + ffn_out)
        return x_proj.squeeze(1)


class HyperbolicFusionGate(nn.Module):
    def __init__(self, static_dim: int, temporal_dim: int, curvature: float = -1.0, dropout: float = 0.1):
        super().__init__()
        self.curvature = curvature
        if static_dim != temporal_dim:
            self.temporal_proj = nn.Linear(temporal_dim, static_dim)
        else:
            self.temporal_proj = nn.Identity()
        self.gate = nn.Sequential(
            nn.Linear(static_dim * 2 + 1, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 2), nn.Softmax(dim=-1),
        )

    def forward(self, static: torch.Tensor, temporal: torch.Tensor,
                h_all_proj: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        temporal = self.temporal_proj(temporal)
        static_h = PoincareBall.expmap0(static, self.curvature)
        temporal_h = PoincareBall.expmap0(temporal, self.curvature)
        
        # v3+ Lorentzian/Hyperbolic Distance based scoring
        # Instead of simple concat MLP, we use hyperbolic distance to define fusion weights
        d_st = PoincareBall.dist(static_h, temporal_h, self.curvature) # (B, 1)
        # Higher distance -> More weight to temporal path as static might be insufficient
        weights = self.gate(torch.cat([static, temporal, d_st], dim=-1))
        
        s_norm = (1 - self.curvature * static_h.norm(dim=-1, keepdim=True).pow(2)).clamp(min=1e-8)
        t_norm = (1 - self.curvature * temporal_h.norm(dim=-1, keepdim=True).pow(2)).clamp(min=1e-8)
        interp = (weights[:, 0:1] * s_norm * static_h + weights[:, 1:2] * t_norm * temporal_h)
        fused_h = interp / (weights[:, 0:1] * s_norm + weights[:, 1:2] * t_norm).clamp(min=1e-8)
        fused = PoincareBall.logmap0(PoincareBall.radius_clamp(fused_h, max_r=0.95, c=self.curvature), self.curvature)
        return fused, weights


class EvidentialHazardHead(nn.Module):
    def __init__(self, input_dim: int, n_hazard: int = 3, lambda_kl: float = 0.1, smoothing: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2), nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 2), nn.ReLU(),
            nn.Linear(input_dim // 2, n_hazard),
        )
        self.lambda_kl = lambda_kl
        self.smoothing = smoothing

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.net(x)
        alpha_pos = F.softplus(logits) + 1.0
        alpha_0 = alpha_pos + 1.0
        expected_prob = alpha_pos / alpha_0
        epistemic_var = alpha_pos / (alpha_0 ** 2 * (alpha_0 + 1))
        return expected_prob, alpha_pos, epistemic_var

    def loss(self, alpha_pos: torch.Tensor, targets: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
        alpha_neg = 1.0
        alpha_0 = alpha_pos + alpha_neg
        targets = targets.clamp(min=0)
        d0 = torch.digamma(alpha_0)
        d1 = torch.digamma(alpha_pos)
        d2 = torch.digamma(torch.full_like(alpha_pos, alpha_neg))
        loss_data = targets * (d0 - d1) + (1 - targets) * (d0 - d2)
        kl = (torch.lgamma(alpha_0) - torch.lgamma(alpha_pos) - torch.lgamma(torch.tensor(alpha_neg, device=alpha_pos.device))
              + (alpha_pos - 1) * (d1 - d0) + (alpha_neg - 1) * (d2 - d0))
        return loss_data.mean() + self.lambda_kl * kl.mean()


class UncertaintyProtoRetriever(nn.Module):
    def __init__(self, query_dim: int, n_hazard: int = 3, k: int = 8, ema_alpha: float = 0.992,
                 n_prototypes: int = 768, proto_dim: int = 256):
        super().__init__()
        self.k = k
        self.ema_alpha = ema_alpha
        self.n_prototypes = n_prototypes
        self.proto_dim = proto_dim
        self.n_hazard = n_hazard
        
        self.query_proj = nn.Sequential(
            nn.Linear(query_dim, proto_dim),
            nn.LayerNorm(proto_dim),
            nn.GELU(),
            nn.Linear(proto_dim, proto_dim),
            nn.LayerNorm(proto_dim),
        )
        
        self.register_buffer("prototypes", torch.randn(n_prototypes, proto_dim))
        nn.init.orthogonal_(self.prototypes, gain=0.1)
        self.register_buffer("proto_labels", torch.zeros(n_prototypes, n_hazard))
        self.register_buffer("proto_counts", torch.zeros(n_prototypes))
        self.register_buffer("n_seen", torch.zeros(1))
        
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        # Plana sadık: Diversity regularization weight
        self.diversity_weight = 0.01
        
        self.head = nn.Sequential(
            nn.Linear(proto_dim + n_hazard, proto_dim // 2),
            nn.LayerNorm(proto_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(proto_dim // 2, n_hazard),
        )

    @torch.no_grad()
    def update(self, query: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor | None = None):
        if self.training:
            B = query.size(0)
            z = F.normalize(self.query_proj(query.detach()), dim=-1)
            sim = z @ F.normalize(self.prototypes, dim=-1).T
            _, indices = torch.topk(sim, 1, dim=-1)
            indices = indices.squeeze(-1)
            
            alpha = min(self.ema_alpha, 1.0 - 1.0 / (self.n_seen.item() + 1.0))
            
            for i in range(B):
                idx = indices[i]
                self.prototypes[idx] = alpha * self.prototypes[idx] + (1 - alpha) * z[i]
                if labels is not None:
                    self.proto_labels[idx] = alpha * self.proto_labels[idx] + (1 - alpha) * labels[i].float()
                self.proto_counts[idx] += 1.0
            
            self.n_seen += B

    def diversity_loss(self) -> torch.Tensor:
        # Diversity Regularization: Prototypes should not be too close to each other
        p = F.normalize(self.prototypes, dim=-1)
        sim = p @ p.T
        # Mask diagonal and take mean of off-diagonal similarities
        mask = torch.eye(self.n_prototypes, device=sim.device).bool()
        return sim.masked_select(~mask).pow(2).mean()

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        z = F.normalize(self.query_proj(query), dim=-1)
        sim = z @ F.normalize(self.prototypes, dim=-1).T
        
        effective_k = min(self.k, self.n_prototypes)
        weights, indices = torch.topk(sim, effective_k, dim=-1)
        
        temp = self.temperature.clamp(0.02, 0.5)
        weights = F.softmax(weights / temp, dim=-1)
        
        neighbor_labels = self.proto_labels[indices]
        agg_labels = (weights.unsqueeze(-1) * neighbor_labels).sum(dim=1)
        
        z_ret = (weights.unsqueeze(-1) * F.normalize(self.prototypes[indices], dim=-1)).sum(dim=1)
        
        out = self.head(torch.cat([z_ret, agg_labels], dim=-1))
        return out


class TemporalPriorPredictor(nn.Module):
    """Learns to predict temporal fusion output from static features (cold prior)."""
    def __init__(self, static_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, static_dim),
        )
        self.residual_proj = nn.Linear(static_dim, static_dim)

    def forward(self, z_static: torch.Tensor) -> torch.Tensor:
        return self.net(z_static) + self.residual_proj(z_static)


class CAGradProjector:
    @staticmethod
    def apply(losses: dict[str, torch.Tensor], model: nn.Module, c: float = 0.4) -> torch.Tensor:
        task_names = [n for n, l in losses.items() if l.requires_grad and l.grad_fn is not None]
        if len(task_names) < 2:
            return losses.get("loss_final", next(iter(losses.values())))

        all_params = list(model.parameters())
        per_task = []
        for name in task_names:
            g = torch.autograd.grad(losses[name], all_params, retain_graph=True,
                                    create_graph=False, allow_unused=True)
            per_task.append(g)

        n_params = len(all_params)
        shared = [p_idx for p_idx in range(n_params)
                  if all(g[p_idx] is not None for g in per_task)]
        if len(shared) < 1:
            return losses.get("loss_final", next(iter(losses.values())))

        G_list = []
        for g in per_task:
            task_g = torch.cat([g[p_idx].view(-1) for p_idx in shared])
            G_list.append(task_g)
        G = torch.stack(G_list)

        GG = G @ G.T
        n_tasks = len(task_names)
        x0 = torch.zeros(n_tasks, device=G.device)
        x0[0] = 1.0
        b = GG @ x0
        A = GG + 1e-8 * torch.eye(n_tasks, device=G.device)
        try:
            sol = torch.linalg.solve(A, b)
        except RuntimeError:
            return losses.get("loss_final", next(iter(losses.values())))

        scale = c / (sol.norm().clamp(min=1e-8))
        w = sol * scale
        if torch.isnan(w).any() or torch.isinf(w).any():
            return losses.get("loss_final", next(iter(losses.values())))
        dummy_loss = sum(w[i] * losses[name] for i, name in enumerate(task_names))
        return dummy_loss
