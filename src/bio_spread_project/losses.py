from __future__ import annotations

import torch


def focal_pairwise_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    knownness: torch.Tensor,
    gamma: float = 2.0,
    max_pairs: int = 4096,
) -> torch.Tensor:
    pos_idx = torch.where(labels > 0.5)[0]
    neg_idx = torch.where(labels <= 0.5)[0]
    if pos_idx.numel() == 0 or neg_idx.numel() == 0:
        return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)

    pair_count = min(max_pairs, int(pos_idx.numel() * neg_idx.numel()))
    pos_pick = pos_idx[torch.randint(0, pos_idx.numel(), (pair_count,), device=scores.device)]
    neg_pick = neg_idx[torch.randint(0, neg_idx.numel(), (pair_count,), device=scores.device)]

    s_pos = scores[pos_pick]
    s_neg = scores[neg_pick]
    p_neg = torch.sigmoid(s_neg)

    w = knownness[pos_pick] * knownness[neg_pick]
    w = w / (w.mean().clamp_min(1e-6))

    margin = torch.sigmoid(s_pos - s_neg).clamp(1e-6, 1.0 - 1e-6)
    focal = torch.pow(1.0 - p_neg, gamma)
    return (-(w * focal * torch.log(margin))).mean()


def soft_ndcg_loss(scores: torch.Tensor, labels: torch.Tensor, topk: int = 25) -> torch.Tensor:
    n = scores.shape[0]
    if n <= 1:
        return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)

    s_i = scores.view(-1, 1)
    s_j = scores.view(1, -1)
    # Smooth rank approximation.
    rank = 1.0 + torch.sigmoid(s_j - s_i).sum(dim=1)

    rel = labels.float().clamp_min(0.0)
    gain = torch.pow(2.0, rel) - 1.0
    discount = torch.log2(1.0 + rank)
    dcg = (gain / discount).sum()

    ideal_rel, _ = torch.sort(rel, descending=True)
    ideal_k = min(topk, ideal_rel.shape[0])
    ideal_gain = torch.pow(2.0, ideal_rel[:ideal_k]) - 1.0
    ideal_discount = torch.log2(torch.arange(2, ideal_k + 2, device=scores.device, dtype=scores.dtype))
    idcg = (ideal_gain / ideal_discount).sum().clamp_min(1e-6)

    ndcg = (dcg / idcg).clamp(0.0, 1.5)
    return 1.0 - ndcg
