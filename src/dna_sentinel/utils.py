"""Consolidated utilities: FASTA, Dataset, Metrics, Prediction, Service, Augmentation, and Muon."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from torch.optim import Optimizer

if TYPE_CHECKING:
    from dna_sentinel.model import KmerTransformer

# =====================================================================
# 1. FASTA & SEQUENCE UTILITIES
# =====================================================================

DNA_ALPHABET = frozenset("ACGT")
_RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")

_CANONICAL_TABLE = {i: "N" for i in range(256)}
for char in "ACGT":
    _CANONICAL_TABLE[ord(char)] = char
for char in " \t\r\n-":
    _CANONICAL_TABLE[ord(char)] = None


def canonical_dna(seq: str) -> str:
    upper = seq.upper()
    translated = upper.translate(_CANONICAL_TABLE)
    if any(ord(c) >= 256 for c in translated):
        return "".join(c if c in DNA_ALPHABET else "N" for c in translated)
    return translated


def revcomp(seq: str) -> str:
    return canonical_dna(seq).translate(_RC_TABLE)[::-1]


def read_fasta(path: str | Path) -> Iterator[tuple[str, str]]:
    sid: str | None = None
    chunks: list[str] = []
    with Path(path).open("rt", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            if raw.startswith(">"):
                if sid is not None:
                    yield sid, canonical_dna("".join(chunks))
                sid = raw[1:].split()[0]
                chunks = []
            else:
                chunks.append(raw)
    if sid is not None:
        yield sid, canonical_dna("".join(chunks))


def write_fasta(records: Iterable[tuple[str, str]], path: str | Path, width: int = 80) -> None:
    with Path(path).open("wt", encoding="utf-8") as handle:
        for sid, seq in records:
            dna = canonical_dna(seq)
            handle.write(f">{sid}\n")
            for i in range(0, len(dna), width):
                handle.write(dna[i : i + width] + "\n")


# =====================================================================
# 2. DATASET DEFINITIONS & AUGMENTATIONS
# =====================================================================

@dataclass(frozen=True)
class LabeledSequence:
    sequence_id: str
    dna: str
    mobility: int
    amr: int
    expansion: int

    def clean(self) -> LabeledSequence:
        return LabeledSequence(self.sequence_id, canonical_dna(self.dna), int(self.mobility), int(self.amr), int(self.expansion))


def save_jsonl(records: list[LabeledSequence], path: str | Path) -> None:
    with Path(path).open("wt", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(asdict(rec.clean()), sort_keys=True) + "\n")


def load_jsonl(path: str | Path) -> list[LabeledSequence]:
    records: list[LabeledSequence] = []
    with Path(path).open("rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(LabeledSequence(**json.loads(line)))
    return records


def rc_augment(records: list[LabeledSequence]) -> list[LabeledSequence]:
    augmented = []
    for rec in records:
        augmented.append(rec)
        augmented.append(LabeledSequence(
            sequence_id=f"{rec.sequence_id}_rc",
            dna=revcomp(rec.dna),
            mobility=rec.mobility,
            amr=rec.amr,
            expansion=rec.expansion,
        ))
    return augmented


class WindowDropout:
    def __init__(self, drop_rate: float = 0.25):
        self.drop_rate = drop_rate

    def __call__(self, features: torch.Tensor | list[torch.Tensor], mask: torch.Tensor, training: bool = True):
        if not training:
            return features, mask
        B, W = mask.shape[:2]
        keep = (torch.rand((B, W), device=mask.device) >= self.drop_rate).float()
        keep[:, 0] = 1.0
        keep_u = keep.unsqueeze(-1)
        if isinstance(features, list):
            return [feat * keep_u for feat in features], mask & keep.bool()
        return features * keep_u, mask & keep.bool()


# =====================================================================
# 3. EVALUATION METRICS
# =====================================================================

def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    if y.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if mask.any():
            ece += float(mask.mean() * abs(y[mask].mean() - p[mask].mean()))
    return ece


def binary_metrics(y_true: list[float] | np.ndarray, y_prob: list[float] | np.ndarray, prefix: str) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    out: dict[str, float] = {}
    if len(np.unique(y)) < 2:
        out[f"{prefix}_auroc"] = 0.5
        out[f"{prefix}_auprc"] = float(y.mean()) if y.size else 0.0
    else:
        out[f"{prefix}_auroc"] = float(roc_auc_score(y, p))
        out[f"{prefix}_auprc"] = float(average_precision_score(y, p))
    out[f"{prefix}_brier"] = float(brier_score_loss(y, p)) if y.size else 0.0
    out[f"{prefix}_ece"] = expected_calibration_error(y, p)
    return out


def multiclass_metrics(y_true: list[int] | np.ndarray, logits: np.ndarray, prefix: str) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    pred = logits.argmax(axis=1)
    return {
        f"{prefix}_accuracy": float(accuracy_score(y, pred)),
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }


# =====================================================================
# 4. PREDICTION & INFERENCE SERVICE
# =====================================================================

@dataclass(frozen=True)
class Prediction:
    sequence_id: str
    mobility_probs: list[float]
    amr_probability: float
    expansion_probability: float
    risk_score: float
    top_windows: list[dict[str, float]]


@torch.inference_mode()
def predict_one(model: KmerTransformer, sequence_id: str, dna: str, device: str = "cpu", top_k: int = 5) -> Prediction:
    from dna_sentinel.features import MultiScaleKmerConfig, MultiScaleKmerExtractor, window_sequence
    model.to(device)
    model.eval()

    dna = canonical_dna(dna)
    extractor = MultiScaleKmerExtractor(MultiScaleKmerConfig(n_features=model.config.n_kmer_features))
    feat, spec, mask, sid = extractor.extract(dna)

    feat = feat.unsqueeze(0).to(device)
    spec = spec.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    sid = sid.unsqueeze(0).to(device)

    out = model(feat, spec, mask, sid)

    mobility = torch.softmax(out["mobility_logits"], dim=-1).squeeze(0).cpu().tolist()
    amr = float(torch.sigmoid(out["amr_logits"]).item())
    expansion = float(torch.sigmoid(out["expansion_logits"]).item())
    mobile = max(mobility[1], mobility[2])
    risk = float(((mobile**2 + amr**2 + expansion**2) / 3.0)**0.5)

    weights = out["evidence_weights"].squeeze(0).cpu()
    active_mask = mask.squeeze(0).cpu().bool()

    all_windows_info = []
    for ws, st, mw in zip(extractor.config.window_sizes, extractor.config.strides, extractor.config.max_windows):
        windows = window_sequence(dna, ws, st, mw)
        for i in range(mw):
            if i < len(windows):
                w = windows[i]
                start = i * st
                all_windows_info.append({"start": float(start), "end": float(start + len(w))})
            else:
                all_windows_info.append({"start": 0.0, "end": 0.0})

    sorted_indices = torch.argsort(weights, descending=True)
    top_windows = []
    for idx in sorted_indices.tolist():
        if len(top_windows) >= top_k:
            break
        if active_mask[idx]:
            info = all_windows_info[idx]
            top_windows.append({"start": info["start"], "end": info["end"], "weight": float(weights[idx])})

    return Prediction(sequence_id, mobility, amr, expansion, risk, top_windows)


class InferenceService:
    def __init__(self, checkpoint: str, device: str = "cpu") -> None:
        from dna_sentinel.model import KmerTransformer
        self.device = device
        self.checkpoint_path = Path(checkpoint)
        self.model = KmerTransformer.load(self.checkpoint_path, device=device)
        self.model.to(device)
        self.model.eval()

    def predict(self, sequence_id: str, dna: str) -> dict:
        pred = predict_one(self.model, sequence_id, dna, device=self.device)
        return asdict(pred)


# =====================================================================
# 5. MUON OPTIMIZER
# =====================================================================

class Muon(Optimizer):
    def __init__(self, params, lr=1e-4, momentum=0.9, ns_steps=5, adamw_lr=3e-4, adamw_betas=(0.9, 0.95), adamw_wd=0.01):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            ns_steps=ns_steps,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_wd=adamw_wd
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if p.ndim == 2 and min(p.shape) >= 16:
                    if "momentum" not in state:
                        state["momentum"] = torch.zeros_like(p.grad)
                    buf = state["momentum"]
                    buf.mul_(group["momentum"]).add_(p.grad)

                    G = buf
                    M, N = G.shape
                    X = G / G.norm().clamp_min(1e-8)
                    for _ in range(group["ns_steps"]):
                        if M < N:
                            A = X @ X.t()
                            X = 0.5 * (3.0 * torch.eye(M, device=X.device, dtype=X.dtype) - A) @ X
                        else:
                            A = X.t() @ X
                            X = 0.5 * X @ (3.0 * torch.eye(N, device=X.device, dtype=X.dtype) - A)
                    p.add_(X, alpha=-group["lr"])
                else:
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    state["step"] += 1
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    beta1, beta2 = group["adamw_betas"]

                    if group["adamw_wd"] > 0:
                        p.mul_(1.0 - group["adamw_lr"] * group["adamw_wd"])

                    exp_avg.mul_(beta1).add_(p.grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1.0 - beta2)

                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(1e-8)
                    step_size = group["adamw_lr"] / bias_correction1
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
