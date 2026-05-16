from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Generator

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        static_dim: int = 12,
        tax_embed_dim: int = 40,
        cat_embed_dim: int = 0,
        hidden_dim: int = 256,
        n_hazard: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()
        input_dim = static_dim + tax_embed_dim + cat_embed_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim // 2, n_hazard)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.head(h)


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
            total += loss / (2 * sigma) + 0.5 * self.log_sigmas[i]
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
