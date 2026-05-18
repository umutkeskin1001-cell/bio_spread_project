from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from dna_sentinel.fasta import canonical_dna
from dna_sentinel.model import DnaSentinel
from dna_sentinel.tokenizer import DnaTokenizer, window_sequence


@dataclass(frozen=True)
class Prediction:
    sequence_id: str
    mobility_probs: list[float]
    amr_probability: float
    expansion_probability: float
    risk_score: float
    top_windows: list[dict[str, float]]


@torch.inference_mode()
def predict_one(model: DnaSentinel, sequence_id: str, dna: str, device: str = "cpu", top_k: int = 5) -> Prediction:
    model.to(device)
    model.eval()
    cfg = model.cfg
    tok = DnaTokenizer()
    tokens, mask = tok.batch_windows([canonical_dna(dna)], cfg.window_size, cfg.stride, cfg.max_windows)
    out = model(tokens.to(device), mask.to(device))
    mobility = torch.softmax(out.mobility_logits, dim=-1).squeeze(0).cpu().tolist()
    amr = float(torch.sigmoid(out.amr_logits).item())
    expansion = float(torch.sigmoid(out.expansion_logits).item())
    mobile = max(mobility[1], mobility[2])
    risk = float(math.prod([max(1e-6, mobile), max(1e-6, amr), max(1e-6, expansion)]) ** (1 / 3))
    weights = out.evidence_weights.squeeze(0).cpu()
    windows = window_sequence(dna, cfg.window_size, cfg.stride, cfg.max_windows)
    top = torch.topk(weights, k=min(top_k, len(windows))).indices.tolist()
    top_windows = []
    for idx in top:
        start = idx * cfg.stride
        top_windows.append({"start": float(start), "end": float(start + len(windows[idx])), "weight": float(weights[idx])})
    return Prediction(sequence_id, mobility, amr, expansion, risk, top_windows)
