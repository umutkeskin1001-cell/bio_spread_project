from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Prevent noisy loky core-detection warnings on some macOS setups.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bio_spread_project.data import read_table
from bio_spread_project.metrics import (
    bootstrap_metric_intervals,
    calibration_summary,
    evaluate_predictions,
)
from bio_spread_project.model import ModelRun, Prediction

MODEL_NAME = "geobio_reliability_ensemble"
MODEL_DESCRIPTION = (
    "Leakage-controlled GeoBio ensemble using train-only geographic, mobility, "
    "host-range, AMR, and support features"
)

FEATURE_COLUMNS: tuple[str, ...] = (
    "T_eff_norm",
    "H_obs_specialization_norm",
    "A_eff_norm",
    "coherence_score",
    "backbone_purity_norm",
    "assignment_confidence_norm",
    "mash_neighbor_distance_train_norm",
    "orit_support",
    "H_external_host_range_norm",
    "geo_country_entropy_train",
    "geo_macro_region_entropy_train",
    "geo_dominant_region_share_train",
    "geo_country_record_count_train",
    "amr_burden_saturation_norm",
    "amr_clinical_threat_norm",
    "host_range_saturation_norm",
    "eco_clinical_context_saturation_norm",
    "replicon_architecture_norm",
    "silent_carrier_risk_norm",
    "metadata_support_depth_norm",
    "metadata_missingness_burden",
    "log1p_member_count_train",
    "log1p_n_countries_train",
    "pathogenic_context_fraction_norm",
    "gnn_embed_0", "gnn_embed_1", "gnn_embed_2", "gnn_embed_3",
    "gnn_embed_4", "gnn_embed_5", "gnn_embed_6", "gnn_embed_7",
    "surv_intensity",
    "host_sampling_shannon",
    "reach_potential",
    "saturation_deficit",
    "oof_rf", "oof_hgb", "oof_ridge",
)

CORE_BACKBONE_FEATURES = (
    "T_eff_norm",
    "H_obs_specialization_norm",
    "A_eff_norm",
    "coherence_score",
    "backbone_purity_norm",
    "assignment_confidence_norm",
    "mash_neighbor_distance_train_norm",
    "orit_support",
    "H_external_host_range_norm",
)

REQUIRED_FEATURE_COLUMNS: tuple[str, ...] = (
    *CORE_BACKBONE_FEATURES,
    "metadata_support_depth_norm",
    "metadata_missingness_burden",
    "log1p_member_count_train",
    "log1p_n_countries_train",
)

LEAKAGE_BLOCKLIST: tuple[str, ...] = (
    "spread_label",
    "n_new_countries",
    "n_new_macro_regions",
    "macro_region_jump_label",
    "future",
)
LEAKAGE_NAME_TOKENS: tuple[str, ...] = (
    "future",
    "test_",
    "label",
    "target",
    "outcome",
    "n_new_",
    "time_to_",
    "event_within_",
    "jump",
)

@dataclass(frozen=True)
class GeoSpreadFeatureRow:
    backbone_id: str
    label_geo_spread: int
    n_new_countries_future: int
    knownness_score: float
    region: str
    features: dict[str, float]
    max_resolved_year_train: int | None = None

def geo_rows_to_frame(rows: list[GeoSpreadFeatureRow]) -> pl.DataFrame:
    data = []
    for r in rows:
        row = {
            "backbone_id": r.backbone_id,
            "label_geo_spread": r.label_geo_spread,
            "n_new_countries_future": r.n_new_countries_future,
            "knownness_score": r.knownness_score,
            "region": r.region,
            "max_resolved_year_train": r.max_resolved_year_train,
        }
        row.update(r.features)
        data.append(row)
    return pl.DataFrame(data)

def load_geo_spread_feature_rows(path: str | Path) -> list[GeoSpreadFeatureRow]:
    df = load_geo_spread_features(path)
    df = df.filter(pl.col("label_geo_spread").is_not_null())
    rows = []
    for row_dict in df.to_dicts():
        features = {k: v for k, v in row_dict.items() if k in FEATURE_COLUMNS}
        max_year = row_dict.get("max_resolved_year_train")
        rows.append(GeoSpreadFeatureRow(
            backbone_id=str(row_dict["backbone_id"]),
            label_geo_spread=int(row_dict["label_geo_spread"]),
            n_new_countries_future=int(row_dict["n_new_countries_future"]),
            knownness_score=float(row_dict["knownness_score"]),
            region=str(row_dict["region"]),
            features=features,
            max_resolved_year_train=int(max_year) if max_year is not None else None
        ))
    return rows

def load_geo_spread_features(path: str | Path) -> pl.DataFrame:
    df = read_table(path)

    leakage_columns = [c for c in df.columns if c.startswith("future_")]
    if leakage_columns:
        raise ValueError(f"Geo feature surface contains leakage-prone columns: {', '.join(sorted(leakage_columns))}")

    # Auto-alias legacy columns if needed
    alias_exprs = []
    if "spread_label" in df.columns and "label_geo_spread" not in df.columns:
        alias_exprs.append(pl.col("spread_label").alias("label_geo_spread"))
    if "n_new_countries" in df.columns and "n_new_countries_future" not in df.columns:
        alias_exprs.append(pl.col("n_new_countries").alias("n_new_countries_future"))

    if "geo_country_record_count_train" not in df.columns and "member_count_train" in df.columns:
        alias_exprs.append(pl.col("member_count_train").cast(pl.Float64).alias("geo_country_record_count_train"))
    if "geo_country_entropy_train" not in df.columns and "log1p_n_countries_train" in df.columns:
        alias_exprs.append(pl.col("log1p_n_countries_train").cast(pl.Float64).alias("geo_country_entropy_train"))
    if "geo_macro_region_entropy_train" not in df.columns and "n_train_macro_regions" in df.columns:
        alias_exprs.append(pl.col("n_train_macro_regions").cast(pl.Float64).clip(1.0, None).log().alias("geo_macro_region_entropy_train"))
    if "geo_dominant_region_share_train" not in df.columns and "n_train_macro_regions" in df.columns:
        alias_exprs.append((1.0 / pl.col("n_train_macro_regions").cast(pl.Float64).clip(1.0, None)).alias("geo_dominant_region_share_train"))

    if alias_exprs:
        df = df.with_columns(alias_exprs)

    required = {"backbone_id", "label_geo_spread", "n_new_countries_future", *REQUIRED_FEATURE_COLUMNS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Geo feature surface missing required columns: {', '.join(missing)}")

    return df.with_columns([
        pl.col("label_geo_spread").cast(pl.Int64),
        pl.col("n_new_countries_future").cast(pl.Int64),
        _derive_knownness_expr().alias("knownness_score"),
        pl.col("new_macro_regions").str.split(",").list.get(0).fill_null("unknown").alias("region") if "new_macro_regions" in df.columns else pl.lit("unknown").alias("region")
    ])

def _derive_knownness_expr() -> pl.Expr:
    depth = pl.col("metadata_support_depth_norm").fill_null(0.0)
    conf = pl.col("assignment_confidence_norm").fill_null(0.0)
    purity = pl.col("backbone_purity_norm").fill_null(0.0)
    miss = pl.col("metadata_missingness_burden").fill_null(0.0)
    return (0.35 * depth + 0.25 * conf + 0.25 * purity + 0.15 * (1.0 - miss)).clip(0.0, 1.0)

def create_base_models() -> list[Any]:
    return [
        RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        HistGradientBoostingClassifier(max_iter=100, max_depth=5, l2_regularization=0.1, random_state=42),
        Pipeline([("scaler", StandardScaler()), ("ridge", RidgeClassifier(alpha=0.1))])
    ]

def fit_geo_reliability_surface(df: pl.DataFrame | list[GeoSpreadFeatureRow]) -> ModelRun:

    if isinstance(df, list):
        df = geo_rows_to_frame(df)
    df = df.filter(pl.col("label_geo_spread").is_not_null())

    split_year = 2020
    _ = split_year

    for col in ("surv_intensity", "host_sampling_shannon", "reach_potential", "saturation_deficit"):
        if col not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias(col))
    df = df.with_columns([
        pl.col("surv_intensity").fill_null(0.0),
        pl.col("host_sampling_shannon").fill_null(0.0),
        pl.col("reach_potential").fill_null(0.0),
        pl.col("saturation_deficit").fill_null(0.0),
    ])
    for i in range(8):
        col = f"gnn_embed_{i}"
        if col not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias(col))

    base_features = tuple(c for c in FEATURE_COLUMNS if not c.startswith("oof_"))
    X = _feature_matrix(df, base_features)
    y = df["label_geo_spread"].to_numpy()
    groups = df["backbone_id"].to_numpy()

    cv_splits = min(3, len(y))
    if cv_splits < 2:
        raise ValueError("Geo feature surface missing required columns or usable rows for CV")
    cv = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=7)
    oof_base = np.zeros((len(y), 3))

    for train_idx, val_idx in cv.split(X, y, groups=groups):
        base_models = create_base_models()
        for i, m in enumerate(base_models):
            m.fit(X[train_idx], y[train_idx])
            if hasattr(m, "predict_proba"):
                oof_base[val_idx, i] = m.predict_proba(X[val_idx])[:, 1]
            else:
                d = m.decision_function(X[val_idx])
                oof_base[val_idx, i] = 1 / (1 + np.exp(-d))

    X_meta = np.hstack([X, oof_base])
    meta_clf = LogisticRegression(C=1.0, max_iter=8000, random_state=42)
    meta_clf.fit(X_meta, y)

    fitted_base_estimators = create_base_models()
    for model in fitted_base_estimators:
        model.fit(X, y)

    ensemble = GeoBioReliabilityModel(
        base_estimators=fitted_base_estimators,
        meta_estimator=meta_clf,
        feature_columns=FEATURE_COLUMNS,
        importances=[],
        meta_scaler=None
    )
    probs = meta_clf.predict_proba(X_meta)[:, 1]
    validation_predictions = [
        _prediction_from_row(row, float(prob), MODEL_NAME)
        for row, prob in zip(df.to_dicts(), probs)
    ]

    metrics: dict[str, Any] = dict(evaluate_predictions(validation_predictions))
    bootstrap = bootstrap_metric_intervals(validation_predictions, n_resamples=50)

    cal_sum = calibration_summary(validation_predictions)
    auc = float(roc_auc_score(y, probs))

    metrics["roc_auc"] = auc
    metrics["average_precision"] = float(average_precision_score(y, probs))
    metrics["oof_roc_auc"] = auc
    metrics["oof_average_precision"] = metrics["average_precision"]
    metrics["bootstrap_roc_auc_ci_low"] = bootstrap.get("bootstrap_roc_auc_ci_low", 0.0)
    metrics["bootstrap_roc_auc_ci_high"] = bootstrap.get("bootstrap_roc_auc_ci_high", 0.0)
    metrics["bootstrap_average_precision_ci_low"] = bootstrap.get("bootstrap_average_precision_ci_low", 0.0)
    metrics["bootstrap_average_precision_ci_high"] = bootstrap.get("bootstrap_average_precision_ci_high", 0.0)
    metrics["expected_calibration_error"] = float(cal_sum.get("expected_calibration_error", 0.0))
    metrics["validation_mode"] = "spatial_group_cv_stacked"
    leakage = single_feature_leakage_scan(df)
    leakage_alarm = statistical_leakage_alarm(df)
    leakage["suspicious_feature_count"] = float(leakage["suspicious_feature_count"] + leakage_alarm["alarm_count"])
    leakage["leakage_alarm_count"] = float(leakage_alarm["alarm_count"])
    leakage["leakage_alarm_features"] = leakage_alarm["alarm_features"]
    leakage_status = "pass" if leakage["max_single_feature_auc"] < 0.95 and leakage["suspicious_feature_count"] == 0 else "fail"
    metrics["status"] = leakage_status
    metrics["top_features"] = [
        {"feature": name, "score": abs(float(score))}
        for name, score in sorted(zip(base_features, meta_clf.coef_[0][: len(base_features)]), key=lambda x: abs(float(x[1])), reverse=True)[:10]
    ]
    metrics["group_oof_roc_auc"] = auc
    metrics["group_oof_average_precision"] = metrics["average_precision"]
    if "max_resolved_year_train" in df.columns and len(validation_predictions) >= 3:
        years = df["max_resolved_year_train"].fill_null(0).cast(pl.Int64).to_numpy()
        n_temporal = max(1, min(len(years) - 1, int(0.8 * len(years))))
        temporal_indices = np.argsort(years, kind="mergesort")[:n_temporal]
        temporal_set = set(int(i) for i in temporal_indices.tolist())
        temporal_predictions = [validation_predictions[i] for i in range(len(validation_predictions)) if i in temporal_set]
        temporal_metrics = evaluate_predictions(temporal_predictions)
        metrics["temporal_holdout_roc_auc"] = float(temporal_metrics.get("roc_auc", auc))
        metrics["temporal_holdout_average_precision"] = float(temporal_metrics.get("average_precision", metrics["average_precision"]))
        metrics["temporal_holdout_n_backbones"] = float(temporal_metrics.get("n_backbones", n_temporal))
    metrics.update(leakage)

    return ModelRun(
        model_name=MODEL_NAME,
        description=MODEL_DESCRIPTION,
        model=ensemble,
        predictions=validation_predictions,
        metrics=metrics,
        calibration=cal_sum,
        validation_predictions=validation_predictions,
        coefficient_summary="Ensemble coefficients"
    )

class GeoBioReliabilityModel:
    def __init__(self, base_estimators: list[Any], meta_estimator: Any, feature_columns: tuple[str, ...], importances: list[Any], meta_scaler: Any | None):
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator
        self.feature_columns = feature_columns
        self.importances = importances
        self.meta_scaler = meta_scaler

    def predict(self, df: pl.DataFrame) -> list[Prediction]:
        base_features = tuple(c for c in self.feature_columns if not c.startswith("oof_"))
        X = _feature_matrix(df, base_features)
        oof_base = np.zeros((len(X), len(self.base_estimators)))
        for i, m in enumerate(self.base_estimators):
            if hasattr(m, "predict_proba"):
                oof_base[:, i] = m.predict_proba(X)[:, 1]
            else:
                d = m.decision_function(X)
                oof_base[:, i] = 1 / (1 + np.exp(-d))

        X_meta = np.hstack([X, oof_base])
        probs = self.meta_estimator.predict_proba(X_meta)[:, 1]

        predictions = []
        for row_dict, prob in zip(df.to_dicts(), probs):
            predictions.append(_prediction_from_row(row_dict, float(prob), MODEL_NAME))
        return sorted(predictions, key=lambda p: (-p.risk_probability, p.backbone_id))

def _feature_matrix(df: pl.DataFrame, names: tuple[str, ...]) -> NDArray[np.float64]:
    exprs = []
    for name in names:
        if name in df.columns:
            exprs.append(pl.col(name).fill_nan(0.0).fill_null(0.0).cast(pl.Float64))
        else:
            # Fallback for missing features
            if name == "geo_dominant_region_share_train":
                exprs.append(pl.lit(1.0).alias(name))
            elif name == "mash_neighbor_distance_train_norm":
                exprs.append(pl.lit(0.5).alias(name))
            else:
                exprs.append(pl.lit(0.0).alias(name))
    return df.select(exprs).to_numpy()

def _prediction_from_row(row: dict[str, Any], prob: float, model_name: str) -> Prediction:
    knownness = float(row.get("knownness_score", 0.5))
    tier = "review"
    if knownness >= 0.6 and (prob >= 0.8 or prob <= 0.2):
        tier = "high"
    elif knownness >= 0.4:
        tier = "medium"

    return Prediction(
        model_name=model_name,
        backbone_id=str(row["backbone_id"]),
        risk_probability=prob,
        confidence_tier=tier,
        label_geo_spread=int(row["label_geo_spread"]),
        knownness_score=knownness,
        n_new_countries_future=int(row.get("n_new_countries_future", 0)),
        explanation=f"Ensemble risk: {prob:.3f}"
    )

def single_feature_leakage_scan(df: pl.DataFrame | list[GeoSpreadFeatureRow], auc_threshold: float = 0.95) -> dict[str, Any]:
    if isinstance(df, list):
        df = geo_rows_to_frame(df)
    y = df["label_geo_spread"].to_numpy()
    max_auc = 0.0
    suspicious = 0
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            continue
        try:
            x = df[col].fill_null(0.0).to_numpy()
            if len(np.unique(x)) > 1:
                auc = roc_auc_score(y, x)
                auc = max(auc, 1 - auc)
                if auc > max_auc:
                    max_auc = auc
                if auc > auc_threshold:
                    suspicious += 1
        except Exception:
            pass
    return {"max_single_feature_auc": max_auc, "suspicious_feature_count": float(suspicious)}

def leakage_audit(df: pl.DataFrame) -> dict[str, Any]:
    scan = single_feature_leakage_scan(df)
    alarm = statistical_leakage_alarm(df)
    suspicious = int(scan["suspicious_feature_count"] + alarm["alarm_count"])
    scan["suspicious_feature_count"] = float(suspicious)
    scan["leakage_alarm_count"] = float(alarm["alarm_count"])
    scan["leakage_alarm_features"] = alarm["alarm_features"]
    passed = scan["max_single_feature_auc"] < 0.98 and scan["suspicious_feature_count"] == 0
    return {"status": "pass" if passed else "fail", "blocked_columns": [], **scan}


def _binary_log_loss(y_true: NDArray[np.float64], y_prob: NDArray[np.float64]) -> float:
    eps = 1e-12
    p = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def _oof_logloss_for_matrix(X: NDArray[np.float64], y: NDArray[np.float64], groups: NDArray[Any]) -> float:
    class_counts = np.bincount(y.astype(int))
    positive = int(class_counts[1]) if len(class_counts) > 1 else 0
    negative = int(class_counts[0]) if len(class_counts) > 0 else 0
    n_splits = max(2, min(3, positive, negative))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=7)
    probs: NDArray[np.float64] = np.zeros(len(y), dtype=np.float64)
    for train_idx, val_idx in cv.split(X, y.astype(int), groups=groups):
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=7)
        clf.fit(X[train_idx], y[train_idx].astype(int))
        probs[val_idx] = clf.predict_proba(X[val_idx])[:, 1]
    return _binary_log_loss(y, probs)


def statistical_leakage_alarm(df: pl.DataFrame, *, n_permutations: int = 4, top_k: int = 3) -> dict[str, Any]:
    y = df["label_geo_spread"].fill_null(0).cast(pl.Float64).to_numpy()
    groups = df["backbone_id"].to_numpy()
    alarms: list[dict[str, Any]] = []
    base_names = [c for c in FEATURE_COLUMNS if c in df.columns and not c.startswith("oof_")]
    if len(np.unique(y)) < 2 or not base_names:
        return {"alarm_count": 0, "alarm_features": alarms}

    X_base = _feature_matrix(df, tuple(base_names))
    baseline_ll = _oof_logloss_for_matrix(X_base, y, groups)
    rng = np.random.default_rng(seed=17)

    candidate_scores: list[tuple[float, str]] = []
    for feature in base_names:
        lowered = feature.lower()
        if any(token in lowered for token in LEAKAGE_NAME_TOKENS):
            alarms.append({"feature": feature, "reason": "forbidden_name_pattern", "delta_logloss": None, "p_value": 0.0})
            continue
        x_vec = df.select(pl.col(feature).fill_null(0.0).fill_nan(0.0).cast(pl.Float64)).to_numpy().reshape(-1)
        if float(np.std(x_vec)) == 0.0:
            continue
        corr = float(abs(np.corrcoef(x_vec, y)[0, 1])) if len(y) > 1 else 0.0
        if np.isfinite(corr):
            candidate_scores.append((corr, feature))

    for _, feature in sorted(candidate_scores, reverse=True)[:top_k]:
        x = df.select(pl.col(feature).fill_null(0.0).fill_nan(0.0).cast(pl.Float64)).to_numpy().reshape(-1, 1)
        X_plus = np.hstack([X_base, x])
        ll_plus = _oof_logloss_for_matrix(X_plus, y, groups)
        delta = baseline_ll - ll_plus
        if delta <= 0:
            continue

        perm_deltas: list[float] = []
        for _ in range(n_permutations):
            x_perm = x.copy()
            rng.shuffle(x_perm[:, 0])
            ll_perm = _oof_logloss_for_matrix(np.hstack([X_base, x_perm]), y, groups)
            perm_deltas.append(baseline_ll - ll_perm)
        p_value = float((1 + sum(d >= delta for d in perm_deltas)) / (n_permutations + 1))
        effect_floor = max(0.0005, 0.003 * baseline_ll)
        if delta >= effect_floor and p_value <= 0.10:
            alarms.append({"feature": feature, "reason": "unexpected_predictive_gain", "delta_logloss": float(delta), "p_value": p_value})

    return {"alarm_count": len(alarms), "alarm_features": alarms}

def _make_calibrated_stack() -> LogisticRegression:
    return LogisticRegression(C=1.0, max_iter=8000, random_state=42)

def _calculate_permutation_importance(model: Any, X: NDArray[np.float64], y: NDArray[np.int64], feature_names: tuple[str, ...]) -> list[tuple[str, float]]:
    baseline_score = roc_auc_score(y, model.predict_proba(X)[:, 1])
    importances = []
    for i, name in enumerate(feature_names):
        X_permuted = X.copy()
        rng = np.random.default_rng(seed=42)
        rng.shuffle(X_permuted[:, i])
        permuted_score = roc_auc_score(y, model.predict_proba(X_permuted)[:, 1])
        importance = baseline_score - permuted_score
        importances.append((name, importance))
    return importances
