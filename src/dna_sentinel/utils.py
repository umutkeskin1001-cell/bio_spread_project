from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)

if TYPE_CHECKING:
    from dna_sentinel.model import Cassiopeia

_TRANSLATE_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")
_A = frozenset("ACGT")
_SKIP = frozenset(" \t\r\n-")


def canonical_dna(seq: str) -> str:
    return "".join(c if c in _A else "N" for c in seq.upper() if c not in _SKIP)


def revcomp(seq: str) -> str:
    return canonical_dna(seq).translate(_TRANSLATE_TABLE)[::-1]


def read_fasta(path: str | Path) -> Iterator[tuple[str, str]]:
    sid, chunks = None, []
    with Path(path).open("rt", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            if raw.startswith(">"):
                if sid is not None:
                    yield sid, canonical_dna("".join(chunks))
                sid = raw[1:].split()[0]
                chunks = []
            else:
                chunks.append(raw.rstrip("\n\r"))
    if sid is not None:
        yield sid, canonical_dna("".join(chunks))


def write_fasta(records: Iterable[tuple[str, str]], path: str | Path, width: int = 80) -> None:
    with Path(path).open("wt", encoding="utf-8") as f:
        for sid, seq in records:
            dna = canonical_dna(seq)
            f.write(f">{sid}\n")
            for i in range(0, len(dna), width):
                f.write(f"{dna[i:i+width]}\n")


@dataclass(frozen=True)
class LabeledSequence:
    sequence_id: str
    dna: str
    mobility: int
    amr: int
    expansion: int


def save_jsonl(records: list[LabeledSequence], path: str | Path) -> None:
    with Path(path).open("wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), sort_keys=True) + "\n")


def load_jsonl(path: str | Path) -> list[LabeledSequence]:
    return [LabeledSequence(**json.loads(line))
            for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


class WindowDropout:
    def __init__(self, drop_rate: float = 0.10):
        self.drop_rate = drop_rate

    def __call__(self, feat: torch.Tensor, mask: torch.Tensor, training: bool = True):
        if not training:
            return feat, mask
        B = mask.shape[0]
        keep = torch.rand(B, mask.shape[1], device=mask.device) >= self.drop_rate
        keep = keep | (~keep.any(dim=1, keepdim=True))
        return feat * keep.unsqueeze(-1), mask & keep


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    if y.size == 0:
        return 0.0
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            ece += float(m.mean() * abs(y[m].mean() - p[m].mean()))
    return ece


def task_score(metrics: dict[str, float]) -> float:
    return float((metrics.get("mobility_balanced_accuracy", 0.0)
                  + metrics.get("amr_auroc", 0.0)
                  + metrics.get("expansion_auroc", 0.0)) / 3.0)


def false_positive_summary(mobility_probs: np.ndarray, amr_probs: np.ndarray,
                           expansion_probs: np.ndarray, risk_scores: np.ndarray,
                           threshold: float = 0.5) -> dict[str, float]:
    mob, amr, exp, risk = (np.asarray(p, dtype=float) for p in
                           (mobility_probs, amr_probs, expansion_probs, risk_scores))
    mobile_prob = 1.0 - mob[:, 0]
    return {
        "false_mobile_rate": float((mobile_prob >= threshold).mean()) if mobile_prob.size else 0.0,
        "false_amr_rate": float((amr >= threshold).mean()) if amr.size else 0.0,
        "false_expansion_rate": float((exp >= threshold).mean()) if exp.size else 0.0,
        "risk_q05": float(np.quantile(risk, 0.05)) if risk.size else 0.0,
        "risk_q50": float(np.quantile(risk, 0.50)) if risk.size else 0.0,
        "risk_q95": float(np.quantile(risk, 0.95)) if risk.size else 0.0,
        "risk_mean": float(risk.mean()) if risk.size else 0.0,
    }


def binary_metrics(y_true, y_prob, prefix: str) -> dict[str, float]:
    y, p = np.asarray(y_true, dtype=float), np.nan_to_num(np.asarray(y_prob, dtype=float), nan=0.5)
    out = {}
    if len(np.unique(y)) < 2:
        out[f"{prefix}_auroc"] = 0.5
        out[f"{prefix}_auprc"] = float(y.mean()) if y.size else 0.0
    else:
        out[f"{prefix}_auroc"] = float(roc_auc_score(y, p))
        out[f"{prefix}_auprc"] = float(average_precision_score(y, p))
    out[f"{prefix}_brier"] = float(brier_score_loss(y, p)) if y.size else 0.0
    out[f"{prefix}_ece"] = expected_calibration_error(y, p)
    return out


def multiclass_metrics(y_true, probs: np.ndarray, prefix: str) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    pred = probs.argmax(axis=1)
    return {
        f"{prefix}_accuracy": float(accuracy_score(y, pred)),
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }


@dataclass(frozen=True)
class Prediction:
    sequence_id: str
    mobility_probs: list[float]
    amr_probability: float
    expansion_probability: float
    risk_score: float
    top_windows: list[dict[str, float]]
    top_mobility_windows: list[dict[str, float]]
    top_amr_windows: list[dict[str, float]]
    top_expansion_windows: list[dict[str, float]]


def _top_evidence(scores: torch.Tensor, mask: torch.Tensor, top_k: int) -> list[dict[str, float]]:
    k = min(top_k, int(mask.sum().item()))
    if k <= 0:
        return []
    top = torch.topk(scores + (~mask).float() * (-1e9), k=k).indices
    return [{"window": float(i), "weight": float(scores[i])} for i in top.tolist() if bool(mask[i])]


@torch.inference_mode()
def predict_batch(model: Cassiopeia, sequences: list[tuple[str, str]],
                  device: str = "cpu", top_k: int = 5) -> list[Prediction]:
    from dna_sentinel.features import CanonicalKmerConfig, CanonicalKmerExtractor

    model.to(device).eval()
    if not sequences:
        return []
    ex = CanonicalKmerExtractor(CanonicalKmerConfig())
    feats, structs, masks, scales = [], [], [], []
    for _, dna in sequences:
        f, s, m, sc = ex.extract(canonical_dna(dna))
        feats.append(f)
        structs.append(s)
        masks.append(m)
        scales.append(sc)
    out = model(torch.stack(feats).to(device), torch.stack(masks).to(device),
                struct_features=torch.stack(structs).to(device) if model.has_struct else None,
                scale_ids=torch.stack(scales).to(device))

    mob = torch.softmax(out["mobility_logits"], dim=-1).cpu()
    amr = torch.sigmoid(out["amr_logits"]).cpu()
    exp_raw = out["expansion_logits"].cpu()
    if model.config.expansion_classes > 1:
        exp_prob = torch.softmax(exp_raw, dim=-1)
    else:
        exp_prob = torch.sigmoid(exp_raw)
    ws = out.get("mobility_evidence", out["mob_evidence"]).cpu()
    amr_ev = out.get("amr_evidence", out["mob_evidence"]).cpu()
    exp_ev = out.get("expansion_evidence", out["mob_evidence"]).cpu()
    am = torch.stack(masks).cpu()

    preds = []
    for idx, (seq_id, _) in enumerate(sequences):
        mobile = 1.0 - mob[idx, 0].item()
        exp_val = float(exp_prob[idx, 1]) if model.config.expansion_classes > 1 else float(exp_prob[idx])
        risk = 0.4 * mobile + 0.3 * float(amr[idx]) + 0.3 * exp_val
        wins = _top_evidence(ws[idx], am[idx], top_k)
        mob_wins = _top_evidence(ws[idx], am[idx], top_k)
        amr_wins = _top_evidence(amr_ev[idx], am[idx], top_k)
        exp_wins = _top_evidence(exp_ev[idx], am[idx], top_k)
        preds.append(Prediction(seq_id, mob[idx].tolist(), float(amr[idx]),
                                 exp_val, risk, wins, mob_wins, amr_wins, exp_wins))
    return preds


def predict_one(model: Cassiopeia, sequence_id: str, dna: str,
                device: str = "cpu", top_k: int = 5) -> Prediction:
    return predict_batch(model, [(sequence_id, dna)], device, top_k)[0]


class InferenceService:
    def __init__(self, checkpoint: str, device: str = "cpu") -> None:
        from dna_sentinel.model import Cassiopeia
        self.device = device
        self.model = Cassiopeia.load(checkpoint, device=device)

    def predict(self, sequence_id: str, dna: str) -> dict:
        return asdict(predict_one(self.model, sequence_id, dna, device=self.device))

    def predict_batch(self, sequences: list[Any]) -> list[dict]:
        parsed = [(s["sequence_id"], s["dna"]) if isinstance(s, dict)
                  else s if isinstance(s, tuple) else (s.sequence_id, s.dna)
                  for s in sequences]
        return [asdict(p) for p in predict_batch(self.model, parsed, device=self.device)]
