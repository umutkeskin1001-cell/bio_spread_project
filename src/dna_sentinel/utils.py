from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator

import numpy as np
import torch

if TYPE_CHECKING:
    from dna_sentinel.model import Cassiopeia

logger = logging.getLogger("cassiopeia")

_A = frozenset("ACGT")
_SKIP = frozenset(" \t\r\n-")
_DNA_MAX_LEN = 300_000
_DNA_MIN_LEN = 1


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def configure_logging(level: int = logging.INFO, log_path: str | Path | None = None) -> None:
    root = logging.getLogger("cassiopeia")
    root.setLevel(level)
    root.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    root.addHandler(logging.StreamHandler())
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path))
        fh.setFormatter(fmt)
        root.addHandler(fh)


class CassiopeiaExperiment:
    def __init__(self, name: str, base_dir: str | Path = "experiments", config: dict | None = None):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.dir = Path(base_dir) / f"{ts}_{name}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict] = []
        self._metrics: dict[str, Any] = {}
        self._checkpoint_path: Path | None = None
        if config:
            (self.dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
        configure_logging(log_path=str(self.dir / "run.log"))

    @property
    def checkpoint_path(self) -> Path:
        assert self._checkpoint_path is not None, "no checkpoint saved yet"
        return self._checkpoint_path

    @checkpoint_path.setter
    def checkpoint_path(self, path: str | Path) -> None:
        self._checkpoint_path = Path(path)

    def log_metrics(self, step: int, **kwargs) -> None:
        self.history.append({"step": step, **kwargs})
        (self.dir / "history.json").write_text(json.dumps(self.history, indent=2, default=str))

    def save_report(self, metrics: dict, name: str = "report") -> Path:
        p = self.dir / f"{name}.json"
        p.write_text(json.dumps(metrics, indent=2, default=str))
        self._metrics = metrics
        return p

    def save_checkpoint(self, path: str | Path) -> Path:
        dest = self.dir / Path(path).name
        import shutil

        shutil.copy2(str(path), str(dest))
        self._checkpoint_path = dest
        return dest

    @classmethod
    def compare(cls, *runs: CassiopeiaExperiment) -> dict:
        result = {}
        for run in runs:
            name = run.dir.name
            result[name] = run._metrics
        return result


class ConfigError(Exception):
    pass


class ValidationError(Exception):
    pass


def validate_dna(dna: str, max_len: int = _DNA_MAX_LEN, min_len: int = _DNA_MIN_LEN) -> str:
    cleaned = "".join(c for c in dna.upper() if c not in _SKIP)
    if not cleaned:
        raise ValidationError("empty sequence after cleaning")
    if len(cleaned) > max_len:
        raise ValidationError(f"sequence too long: {len(cleaned)} > {max_len}")
    n_count = sum(1 for c in cleaned if c not in _A)
    if n_count > 0:
        logger.warning("masked %d non-ACGT characters in sequence", n_count)
    return cleaned


_TRANSLATE_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def canonical_dna(seq: str) -> str:
    return "".join(c if c in _A else "N" for c in seq.upper() if c not in _SKIP)


def revcomp(seq: str) -> str:
    return canonical_dna(seq).translate(_TRANSLATE_TABLE)[::-1]


def circular_shift(seq: str, offset: int) -> str:
    dna = canonical_dna(seq)
    if not dna:
        return ""
    step = offset % len(dna)
    return dna[step:] + dna[:step]


def dinuc_shuffle(seq: str, seed: int = 42) -> str:
    """Preserves GC content and dinucleotide frequencies via block shuffling."""
    dna = canonical_dna(seq)
    if len(dna) < 4:
        return dna
    bases = np.array(list(dna))
    rng = np.random.default_rng(seed)
    for _ in range(3):
        pairs = [(bases[i], bases[i + 1]) for i in range(0, len(bases) - 1, 2)]
        rng.shuffle(pairs)
        flat = np.empty(len(bases), dtype='<U1')
        flat[::2] = [p[0] for p in pairs]
        flat[1::2] = [p[1] for p in pairs]
        if len(bases) % 2:
            flat[-1] = bases[-1]
        bases = flat
    return "".join(bases)


def read_fasta(path: str | Path) -> Iterator[tuple[str, str]]:
    sid, chunks = None, []
    with Path(path).open("rt", encoding="utf-8", errors="replace") as f:
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
            if not dna:
                continue
            f.write(f">{sid}\n")
            for i in range(0, len(dna), width):
                f.write(f"{dna[i : i + width]}\n")


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
    return [
        LabeledSequence(**json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class WindowDropout:
    def __init__(self, drop_rate: float = 0.10):
        self.drop_rate = drop_rate

    def __call__(self, feat: torch.Tensor, mask: torch.Tensor, training: bool = True):
        if not training:
            return feat, mask
        keep = torch.rand(mask.shape[0], mask.shape[1], device=mask.device) >= self.drop_rate
        keep = keep | (~keep.any(dim=1, keepdim=True))
        return feat * keep.unsqueeze(-1), mask & keep


class DNASequenceAugmentation:
    def __init__(self, mutation_rate: float = 0.0, truncation_rate: float = 0.0):
        self.mutation_rate = mutation_rate
        self.truncation_rate = truncation_rate

    def __call__(self, records: list[LabeledSequence], training: bool = True):
        if not training or (self.mutation_rate == 0 and self.truncation_rate == 0):
            return records
        result = []
        for r in records:
            dna = r.dna
            if self.mutation_rate > 0 and random.random() < 0.5:
                chars = list(dna)
                n_mut = max(1, int(len(chars) * self.mutation_rate))
                for _ in range(n_mut):
                    idx = random.randrange(len(chars))
                    if chars[idx] in _A:
                        chars[idx] = random.choice([b for b in "ACGT" if b != chars[idx]])
                dna = "".join(chars)
            if self.truncation_rate > 0 and random.random() < self.truncation_rate:
                keep = random.uniform(0.5, 1.0)
                dna = dna[: max(100, int(len(dna) * keep))]
            result.append(LabeledSequence(r.sequence_id, dna, r.mobility, r.amr, r.expansion))
        return result


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
    return float(
        (
            metrics.get("mobility_balanced_accuracy", 0.0)
            + metrics.get("amr_auroc", 0.0)
            + metrics.get("expansion_auroc", 0.0)
        )
        / 3.0
    )


def false_positive_summary(
    mobility_probs: np.ndarray | list,
    amr_probs: np.ndarray | list,
    expansion_probs: np.ndarray | list,
    risk_scores: np.ndarray | list,
    threshold: float = 0.5,
) -> dict[str, float]:
    mob = np.asarray(mobility_probs, dtype=float)
    mobile_prob = 1.0 - (mob[:, 0] if mob.ndim == 2 and mob.shape[1] >= 3 else mob)
    amr = np.asarray(amr_probs, dtype=float).ravel()
    exp = np.asarray(expansion_probs, dtype=float).ravel()
    risk = np.asarray(risk_scores, dtype=float).ravel()
    return {
        "false_mobile_rate": float((mobile_prob >= threshold).mean()) if mobile_prob.size else 0.0,
        "false_amr_rate": float((amr >= threshold).mean()) if amr.size else 0.0,
        "false_expansion_rate": float((exp >= threshold).mean()) if exp.size else 0.0,
        "risk_q05": float(np.quantile(risk, 0.05)) if risk.size else 0.0,
        "risk_q50": float(np.quantile(risk, 0.50)) if risk.size else 0.0,
        "risk_q95": float(np.quantile(risk, 0.95)) if risk.size else 0.0,
        "risk_mean": float(risk.mean()) if risk.size else 0.0,
    }


def compute_risk_score(mob_probs, amr_prob, exp_prob, weights=(0.4, 0.3, 0.3)):
    mobile = 1.0 - mob_probs[0]
    return weights[0] * mobile + weights[1] * amr_prob + weights[2] * exp_prob


def bootstrap_ci(y_true, y_pred, metric_fn, n_resamples: int = 1000, ci: float = 0.95) -> tuple[float, float, float]:
    n = len(y_true)
    rng = np.random.default_rng(42)
    scores = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        try:
            scores.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            continue
    if not scores:
        return float(metric_fn(y_true, y_pred)), 0.0, 0.0
    scores = np.array(scores)
    point = float(metric_fn(y_true, y_pred))
    alpha = (1.0 - ci) / 2.0
    return point, float(np.quantile(scores, alpha)), float(np.quantile(scores, 1.0 - alpha))


def binary_metrics(y_true, y_prob, prefix: str) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

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
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

    y = np.asarray(y_true, dtype=int)
    pred = probs.argmax(axis=1)
    out = {
        f"{prefix}_accuracy": float(accuracy_score(y, pred)),
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }
    n_cls = probs.shape[1]
    cm = confusion_matrix(y, pred, labels=list(range(n_cls)))
    for c in range(n_cls):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        out[f"{prefix}_class{c}_precision"] = float(p)
        out[f"{prefix}_class{c}_recall"] = float(r)
        out[f"{prefix}_class{c}_f1"] = float(2 * p * r / max(1e-8, p + r))
    out[f"{prefix}_confusion_matrix"] = cm.tolist()
    conf = probs.max(axis=1)
    acc_mask = (y == pred).astype(float)
    out[f"{prefix}_ece"] = float(expected_calibration_error(acc_mask, conf, bins=10))
    out[f"{prefix}_brier"] = float(((probs - np.eye(n_cls)[y]) ** 2).sum(axis=1).mean())
    return out


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


_PREDICT_EXTRACTORS: dict[tuple[int, ...], Any] = {}


@torch.inference_mode()
def _predict_pass(model: Cassiopeia, sequences: list[tuple[str, str]], device: str = "cpu") -> dict[str, torch.Tensor]:
    from dna_sentinel.features import CanonicalKmerConfig, CanonicalKmerExtractor, _resolve_max_windows

    model.to(device).eval()
    mw = _resolve_max_windows(model.config.max_windows)
    ns = model.config.n_structural_features
    cache_key = (mw, ns)
    if cache_key not in _PREDICT_EXTRACTORS:
        _PREDICT_EXTRACTORS[cache_key] = CanonicalKmerExtractor(CanonicalKmerConfig(max_windows=mw, n_structural_features=ns))
    ex = _PREDICT_EXTRACTORS[cache_key]
    feats, structs, masks, scales = [], [], [], []
    for _, dna in sequences:
        f, s, m, sc = ex.extract(canonical_dna(dna))
        feats.append(f)
        structs.append(s)
        masks.append(m)
        scales.append(sc)
    feat_s = torch.stack(feats).to(device)
    out = model(
        feat_s,
        torch.stack(masks).to(device),
        struct_features=torch.stack(structs).to(device) if model.has_struct else None,
        scale_ids=torch.stack(scales).to(device),
    )
    mob = torch.softmax(out["mobility_logits"], dim=-1).cpu()
    amr = torch.sigmoid(out["amr_logits"]).cpu()
    exp_raw = out["expansion_logits"].cpu()
    exp_prob = torch.softmax(exp_raw, dim=-1) if model.config.expansion_classes > 1 else torch.sigmoid(exp_raw)
    return {
        "mobility_probs": mob,
        "amr_probs": amr,
        "expansion_probs": exp_prob,
        "mobility_evidence": out["mobility_evidence"].cpu(),
        "amr_evidence": out["amr_evidence"].cpu(),
        "expansion_evidence": out["expansion_evidence"].cpu(),
        "masks": torch.stack(masks).cpu(),
    }


@torch.inference_mode()
def predict_batch(
    model: Cassiopeia, sequences: list[tuple[str, str]], device: str = "cpu",
    top_k: int = 5, rc_average: bool = True, n_circular_shifts: int = 0
) -> list[Prediction]:
    if not sequences:
        return []
    passes = [_predict_pass(model, sequences, device)]
    if rc_average:
        passes.append(
            _predict_pass(model, [(sid, revcomp(dna)) for sid, dna in sequences], device)
        )
    if n_circular_shifts > 0:
        for offset in range(1, n_circular_shifts + 1):
            shifted = [
                (sid, circular_shift(dna, max(1, len(dna)) * offset // (n_circular_shifts + 1)))
                for sid, dna in sequences
            ]
            passes.append(_predict_pass(model, shifted, device))
            if rc_average:
                passes.append(
                    _predict_pass(model, [(sid, revcomp(d)) for sid, d in shifted], device)
                )
    mob = sum(p["mobility_probs"] for p in passes) / len(passes)
    amr = sum(p["amr_probs"] for p in passes) / len(passes)
    exp_prob = sum(p["expansion_probs"] for p in passes) / len(passes)
    am = passes[0]["masks"]
    mob_ev = passes[0]["mobility_evidence"]
    amr_ev = passes[0]["amr_evidence"]
    exp_ev = passes[0]["expansion_evidence"]
    preds = []
    for idx, (seq_id, _) in enumerate(sequences):
        mobile = 1.0 - mob[idx, 0].item()
        exp_val = float(exp_prob[idx, 1]) if model.config.expansion_classes > 1 else float(exp_prob[idx])
        rw = model.config.risk_weights
        risk = rw[0] * mobile + rw[1] * float(amr[idx]) + rw[2] * exp_val
        all_scores = (mob_ev[idx] + amr_ev[idx] + exp_ev[idx]) / 3.0
        preds.append(
            Prediction(
                seq_id,
                mob[idx].tolist(),
                float(amr[idx]),
                exp_val,
                risk,
                _top_evidence(all_scores, am[idx], top_k),
                _top_evidence(mob_ev[idx], am[idx], top_k),
                _top_evidence(amr_ev[idx], am[idx], top_k),
                _top_evidence(exp_ev[idx], am[idx], top_k),
            )
        )
    return preds


def predict_one(
    model: Cassiopeia, sequence_id: str, dna: str, device: str = "cpu", top_k: int = 5, rc_average: bool = True
) -> Prediction:
    return predict_batch(model, [(sequence_id, dna)], device, top_k, rc_average=rc_average)[0]


@torch.inference_mode()
def evaluate_records(
    model: Cassiopeia,
    records: list[LabeledSequence],
    device: str = "cpu",
    batch_size: int = 32,
    top_k: int = 5,
    rc_average: bool = True,
) -> dict[str, float]:
    preds = []
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        preds.extend(
            predict_batch(
                model, [(r.sequence_id, r.dna) for r in chunk], device=device, top_k=top_k, rc_average=rc_average
            )
        )
    mob = np.array([p.mobility_probs for p in preds], dtype=float)
    amr = np.array([p.amr_probability for p in preds], dtype=float)
    exp = np.array([p.expansion_probability for p in preds], dtype=float)
    out = multiclass_metrics(np.array([r.mobility for r in records], dtype=int), mob, "mobility")
    out.update(binary_metrics(np.array([r.amr for r in records], dtype=float), amr, "amr"))
    out.update(binary_metrics(np.array([r.expansion for r in records], dtype=float), exp, "expansion"))
    return out


class InferenceService:
    def __init__(self, checkpoint: str, device: str = "cpu") -> None:
        from dna_sentinel.model import Cassiopeia

        self.device = device
        self.model = Cassiopeia.load(checkpoint, device=device)

    def predict(self, sequence_id: str, dna: str) -> dict:
        return asdict(predict_one(self.model, sequence_id, dna, device=self.device))

    def predict_batch(self, sequences: list[Any]) -> list[dict]:
        parsed = [
            (s["sequence_id"], s["dna"])
            if isinstance(s, dict)
            else s
            if isinstance(s, tuple)
            else (s.sequence_id, s.dna)
            for s in sequences
        ]
        return [asdict(p) for p in predict_batch(self.model, parsed, device=self.device)]
