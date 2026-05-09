from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from bio_spread_project.metrics import compute_metrics
from bio_spread_project.model import Prediction


@dataclass(frozen=True)
class EnhancementSummary:
    enabled: bool
    threshold: float
    low_count: int
    alpha: float
    global_auc_before: float
    global_auc_after: float
    global_ap_before: float
    global_ap_after: float
    low_auc_before: float | None
    low_auc_after: float | None
    low_ap_before: float | None
    low_ap_after: float | None


def _to_arrays(predictions: list[Prediction]) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    y = np.asarray([int(p.label_geo_spread) for p in predictions], dtype=np.int64)
    p = np.asarray([float(p.risk_probability) for p in predictions], dtype=np.float64)
    k = np.asarray([float(p.knownness_score) for p in predictions], dtype=np.float64)
    u = np.asarray([float((p.meta or {}).get("alarm_score", p.alarm_score)) for p in predictions], dtype=np.float64)
    return y, p, k, u


def _slice_metrics(y: NDArray[np.int64], p: NDArray[np.float64]) -> tuple[float | None, float | None]:
    if len(y) < 8 or len(np.unique(y)) < 2:
        return None, None
    m = compute_metrics(labels=y, probabilities=p, bins=5)
    return float(m.roc_auc), float(m.average_precision)


def enhance_low_knownness_predictions(
    predictions: list[Prediction],
    *,
    low_quantile: float = 0.20,
    alpha_max: float = 0.85,
    alpha_penalty: float = 0.03,
    max_global_auc_drop: float = 0.003,
    max_global_ap_drop: float = 0.005,
) -> tuple[list[Prediction], EnhancementSummary]:
    if len(predictions) < 40:
        return predictions, EnhancementSummary(False, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, None, None, None, None)

    y, base_p, knownness, uncertainty = _to_arrays(predictions)
    if len(np.unique(y)) < 2:
        return predictions, EnhancementSummary(False, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, None, None, None, None)

    threshold = float(np.quantile(knownness, low_quantile))
    low_mask = knownness <= threshold
    low_n = int(np.sum(low_mask))
    if low_n < 20 or len(np.unique(y[low_mask])) < 2:
        return predictions, EnhancementSummary(False, threshold, low_n, 0.0, 0.0, 0.0, 0.0, 0.0, None, None, None, None)

    # 1) Slice-aware calibration
    cal_low = IsotonicRegression(out_of_bounds="clip")
    cal_hi = IsotonicRegression(out_of_bounds="clip")
    cal_low.fit(base_p[low_mask], y[low_mask])
    cal_hi.fit(base_p[~low_mask], y[~low_mask])
    p_cal = base_p.copy()
    p_cal[low_mask] = cal_low.transform(base_p[low_mask])
    p_cal[~low_mask] = cal_hi.transform(base_p[~low_mask])

    # 2) Residual expert for low-knownness
    z = np.clip(p_cal, 1e-6, 1 - 1e-6)
    logit = np.log(z / (1.0 - z))
    x_low = np.column_stack(
        [
            logit[low_mask],
            knownness[low_mask],
            1.0 - knownness[low_mask],
            np.clip(uncertainty[low_mask], 0.0, None),
            logit[low_mask] * (1.0 - knownness[low_mask]),
        ]
    )
    residual_model = LogisticRegression(C=0.5, class_weight="balanced", random_state=42, max_iter=1000)
    residual_model.fit(x_low, y[low_mask])
    p_low_hat = residual_model.predict_proba(x_low)[:, 1]
    delta_low = np.zeros_like(base_p)
    delta_low[low_mask] = p_low_hat - p_cal[low_mask]

    # 3) Soft-gated MoE mixing
    gate = 1.0 / (1.0 + np.exp(20.0 * (knownness - threshold)))
    candidate = np.clip(p_cal + gate * delta_low, 1e-6, 1 - 1e-6)

    # 4) Constrained selection (Lagrangian-like search over alpha)
    m_base = compute_metrics(labels=y, probabilities=base_p, bins=5)
    base_auc, base_ap = float(m_base.roc_auc), float(m_base.average_precision)
    low_auc_before, low_ap_before = _slice_metrics(y[low_mask], base_p[low_mask])

    best_alpha = 0.0
    best_score = -1e9
    best_probs = base_p
    alpha_grid = [a for a in (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.85, 0.90, 1.0) if a <= alpha_max + 1e-12]
    for alpha in alpha_grid:
        probs = np.clip((1.0 - alpha) * base_p + alpha * candidate, 1e-6, 1 - 1e-6)
        m = compute_metrics(labels=y, probabilities=probs, bins=5)
        auc_now, ap_now = float(m.roc_auc), float(m.average_precision)
        if auc_now < base_auc - max_global_auc_drop:
            continue
        if ap_now < base_ap - max_global_ap_drop:
            continue
        low_auc_now, low_ap_now = _slice_metrics(y[low_mask], probs[low_mask])
        auc_gain = 0.0 if low_auc_now is None or low_auc_before is None else (low_auc_now - low_auc_before)
        ap_gain = 0.0 if low_ap_now is None or low_ap_before is None else (low_ap_now - low_ap_before)
        score = 2.0 * auc_gain + 1.5 * ap_gain + 0.1 * (auc_now - base_auc) - alpha_penalty * float(alpha)
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)
            best_probs = probs

    m_final = compute_metrics(labels=y, probabilities=best_probs, bins=5)
    low_auc_after, low_ap_after = _slice_metrics(y[low_mask], best_probs[low_mask])

    updated: list[Prediction] = []
    for idx, p in enumerate(predictions):
        meta = dict(p.meta or {})
        meta["low_knownness_enhanced"] = bool(best_alpha > 0.0)
        meta["low_knownness_gate"] = float(gate[idx])
        updated.append(
            Prediction(
                model_name=p.model_name,
                backbone_id=p.backbone_id,
                risk_probability=float(best_probs[idx]),
                confidence_tier=p.confidence_tier,
                label_geo_spread=p.label_geo_spread,
                knownness_score=p.knownness_score,
                n_new_countries_future=p.n_new_countries_future,
                explanation=p.explanation,
                alarm_score=p.alarm_score,
                meta=meta,
            )
        )

    summary = EnhancementSummary(
        enabled=best_alpha > 0.0,
        threshold=threshold,
        low_count=low_n,
        alpha=best_alpha,
        global_auc_before=base_auc,
        global_auc_after=float(m_final.roc_auc),
        global_ap_before=base_ap,
        global_ap_after=float(m_final.average_precision),
        low_auc_before=low_auc_before,
        low_auc_after=low_auc_after,
        low_ap_before=low_ap_before,
        low_ap_after=low_ap_after,
    )
    return updated, summary
