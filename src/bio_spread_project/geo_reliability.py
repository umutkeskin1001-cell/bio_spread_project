from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

# Prevent noisy loky core-detection warnings on some macOS setups.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bio_spread_project.data import read_table
from bio_spread_project.metrics import (
    bootstrap_metric_intervals,
    calibration_summary,
    evaluate_predictions,
)
from bio_spread_project.model import BioSpreadRiskModel, ModelRun, Prediction

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
)

REQUIRED_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    column for column in FEATURE_COLUMNS if not column.startswith("geo_")
)

LEAKAGE_BLOCKLIST: tuple[str, ...] = (
    "label", "future", "test", "n_new", "new_countries",
    "event_within", "time_to", "visibility", "outcome", "severity",
)

ALLOWED_OUTCOME_COLUMNS: frozenset[str] = frozenset(
    {
        "n_countries_test",
        "n_new_countries",
        "test_year_end",
        "min_resolved_year_test",
        "spread_label",
        "visibility_expansion_label",
        "n_new_countries_recomputed",
        "time_to_first_new_country_years",
        "time_to_third_new_country_years",
        "event_within_1y_label",
        "event_within_3y_label",
        "event_within_5y_label",
        "three_countries_within_3y_label",
        "three_countries_within_5y_label",
        "spread_severity_bin",
        "n_test_macro_regions",
        "n_new_macro_regions",
        "new_macro_regions",
        "macro_region_jump_label",
        "n_test_records_seen_in_training",
        "test_seen_in_training_fraction",
        "training_only_future_unseen_backbone_flag",
    }
)


@dataclass(frozen=True)
class GeoSpreadFeatureRow:
    backbone_id: str
    label_geo_spread: int
    n_new_countries_future: int
    knownness_score: float
    max_resolved_year_train: int
    features: dict[str, float]


def geo_rows_to_frame(rows: list[GeoSpreadFeatureRow]) -> pl.DataFrame:
    payload: list[dict[str, Any]] = []
    for row in rows:
        item = {
            "backbone_id": row.backbone_id,
            "label_geo_spread": row.label_geo_spread,
            "n_new_countries_future": row.n_new_countries_future,
            "knownness_score": row.knownness_score,
            "max_resolved_year_train": row.max_resolved_year_train,
        }
        item.update(row.features)
        payload.append(item)
    return pl.DataFrame(payload)


class GeoBioReliabilityModel(BioSpreadRiskModel):
    def __init__(self, estimators: list[tuple[float, Any]], feature_columns: tuple[str, ...], importances: list[dict[str, Any]] | None = None):
        self.model_name = MODEL_NAME
        self.description = MODEL_DESCRIPTION
        self.estimators = list(estimators)
        self.feature_columns = tuple(feature_columns)
        self.importances = {i["feature"]: i["score"] for i in (importances or [])}

    def predict_probabilities(self, df: pl.DataFrame) -> NDArray[np.float64]:
        matrix = _feature_matrix(df, self.feature_columns)
        probabilities = np.zeros(len(df), dtype=float)
        for weight, estimator in self.estimators:
            probabilities += weight * estimator.predict_proba(matrix)[:, 1]
        return np.clip(probabilities, 0.0, 1.0)

    def predict(self, df: pl.DataFrame) -> list[Prediction]:
        probabilities = self.predict_probabilities(df)
        uncertainties = self._estimate_uncertainty(df)

        predictions = []
        rows = df.to_dicts()
        for row, prob, unc in zip(rows, probabilities, uncertainties):
            pred = _prediction_from_row(row, float(prob), MODEL_NAME)
            if pred.meta is not None:
                pred.meta.update({
                    "uncertainty_score": float(unc),
                    "attributions": self._get_attributions(row),
                })
            predictions.append(pred)

        return sorted(predictions, key=lambda row: (-row.risk_probability, row.backbone_id))

    def _estimate_uncertainty(self, df: pl.DataFrame) -> NDArray[np.float64]:
        matrix = _feature_matrix(df, self.feature_columns)
        z_scores = np.abs((matrix - np.mean(matrix, axis=0)) / (np.std(matrix, axis=0) + 1e-6))
        return cast(NDArray[np.float64], np.mean(z_scores, axis=1) / 5.0)

    def _get_attributions(self, row: dict[str, Any]) -> dict[str, float]:
        if not self.importances:
            return {f: float(row.get(f, 0.0) * 0.1) for f in self.feature_columns[:3]}
        impacts = {f: float(float(row.get(f, 0.0)) * self.importances.get(f, 0.0)) for f in self.feature_columns}
        sorted_impacts = sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)
        return dict(sorted_impacts[:5])


def load_geo_spread_feature_rows(path: str | Path) -> list[GeoSpreadFeatureRow]:
    df = read_table(path)
    if df.is_empty():
        raise ValueError(f"GeoSpread feature surface is empty: {path}")
    audit = leakage_audit([column for column in df.columns if column not in ALLOWED_OUTCOME_COLUMNS])
    if audit["status"] != "pass":
        blocked = ", ".join(str(column) for column in audit["blocked_columns"])
        raise ValueError(f"GeoSpread feature surface contains leakage-prone columns: {blocked}")

    alias_exprs: list[pl.Expr] = []
    if "geo_country_entropy_train" not in df.columns:
        if "n_countries_train" in df.columns:
            alias_exprs.append(pl.col("n_countries_train").cast(pl.Float64).log1p().alias("geo_country_entropy_train"))
        elif "log1p_n_countries_train" in df.columns:
            alias_exprs.append(pl.col("log1p_n_countries_train").cast(pl.Float64).alias("geo_country_entropy_train"))
    if "geo_macro_region_entropy_train" not in df.columns and "n_train_macro_regions" in df.columns:
        alias_exprs.append(pl.col("n_train_macro_regions").cast(pl.Float64).log1p().alias("geo_macro_region_entropy_train"))
    if "geo_dominant_region_share_train" not in df.columns and "n_train_macro_regions" in df.columns:
        # Leakage firewall: use the uniform train-region lower bound instead of
        # future macro-region outcomes, preserving the feature contract without
        # letting post-split geography backpropagate into the training matrix.
        alias_exprs.append(
            (1.0 / pl.col("n_train_macro_regions").cast(pl.Float64).clip(lower_bound=1.0))
            .clip(0.0, 1.0)
            .alias("geo_dominant_region_share_train")
        )
    if "geo_country_record_count_train" not in df.columns:
        if "member_count_train" in df.columns:
            alias_exprs.append(pl.col("member_count_train").cast(pl.Float64).alias("geo_country_record_count_train"))
        elif "log1p_member_count_train" in df.columns:
            alias_exprs.append(pl.col("log1p_member_count_train").cast(pl.Float64).exp().sub(1.0).alias("geo_country_record_count_train"))
    if "geo_dominant_region_share_train" not in df.columns and not any(
        expr.meta.output_name() == "geo_dominant_region_share_train" for expr in alias_exprs
    ):
        alias_exprs.append(pl.lit(0.5).alias("geo_dominant_region_share_train"))
    if alias_exprs:
        df = df.with_columns(alias_exprs)

    required = {"backbone_id", "spread_label", "n_new_countries", *REQUIRED_FEATURE_COLUMNS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"GeoSpread feature surface is missing required columns: {', '.join(missing)}")

    # Vectorized derivation
    region_expr = (
        pl.col("new_macro_regions").str.split(",").list.get(0).fill_null("unknown")
        if "new_macro_regions" in df.columns
        else pl.lit("unknown")
    )
    df = df.filter(pl.col("spread_label").cast(pl.Float64).is_in([0.0, 1.0])).with_columns([
        pl.col("spread_label").cast(pl.Int32).alias("label_geo_spread"),
        pl.col("n_new_countries").cast(pl.Int32).alias("n_new_countries_future"),
        _derive_knownness_expr().alias("knownness_score"),
        region_expr.alias("region"),
    ])
    rows: list[GeoSpreadFeatureRow] = []
    for row in df.to_dicts():
        features = {c: float(row[c]) for c in FEATURE_COLUMNS if c in row and row[c] is not None}
        rows.append(
            GeoSpreadFeatureRow(
                backbone_id=str(row["backbone_id"]),
                label_geo_spread=int(row["label_geo_spread"]),
                n_new_countries_future=int(row["n_new_countries_future"]),
                knownness_score=float(row["knownness_score"]),
                max_resolved_year_train=int(row.get("max_resolved_year_train", 0) or 0),
                features=features,
            )
        )
    return rows


def _derive_knownness_expr() -> pl.Expr:
    depth = pl.col("metadata_support_depth_norm").fill_null(0.0)
    conf = pl.col("assignment_confidence_norm").fill_null(0.0)
    purity = pl.col("backbone_purity_norm").fill_null(0.0)
    miss = pl.col("metadata_missingness_burden").fill_null(0.0)
    return (0.35 * depth + 0.25 * conf + 0.25 * purity + 0.15 * (1.0 - miss)).clip(0.0, 1.0)


def fit_geo_reliability_surface(df: pl.DataFrame | list[GeoSpreadFeatureRow]) -> ModelRun:
    from bio_spread_project.audit import render_model_card

    if isinstance(df, list):
        df = geo_rows_to_frame(df)
    X = _feature_matrix(df, FEATURE_COLUMNS)
    y = df["label_geo_spread"].to_numpy()
    groups = df["backbone_id"].to_numpy()

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=7)
    oof_probs = np.zeros(len(y))

    for train_idx, val_idx in cv.split(X, y, groups=groups):
        fold_model = _make_calibrated_stack()
        fold_model.fit(X[train_idx], y[train_idx])
        oof_probs[val_idx] = fold_model.predict_proba(X[val_idx])[:, 1]

    calibrated_model = _make_calibrated_stack()
    calibrated_model.fit(X, y)

    importances = _calculate_permutation_importance(calibrated_model, X, y, FEATURE_COLUMNS)
    ensemble = GeoBioReliabilityModel(
        estimators=[(1.0, calibrated_model)],
        feature_columns=FEATURE_COLUMNS,
        importances=importances
    )

    rows_dict = df.to_dicts()
    validation_predictions = [
        _prediction_from_row(rows_dict[i], float(oof_probs[i]), MODEL_NAME)
        for i in range(len(df))
    ]
    predictions = ensemble.predict(df)
    metrics: dict[str, Any] = dict(evaluate_predictions(validation_predictions))
    bootstrap = bootstrap_metric_intervals(validation_predictions, n_resamples=1000)

    temporal_metrics = _temporal_holdout_metrics(df)
    audit_data = {"quality_gates": {}, "validation": metrics}
    model_card_md = render_model_card(audit=audit_data, coefficient_summary=_format_importance(importances[:5]))
    cal_sum = calibration_summary(validation_predictions)

    oof_auc = float(roc_auc_score(y, oof_probs))
    oof_ap = float(average_precision_score(y, oof_probs))

    metrics.update({
        "roc_auc": oof_auc, "average_precision": oof_ap,
        "oof_roc_auc": oof_auc, "oof_average_precision": oof_ap,
        "group_oof_roc_auc": oof_auc, "group_oof_average_precision": oof_ap,
        "bootstrap_roc_auc_ci_low": bootstrap["bootstrap_roc_auc_ci_low"],
        "bootstrap_roc_auc_ci_high": bootstrap["bootstrap_roc_auc_ci_high"],
        "bootstrap_average_precision_ci_low": bootstrap["bootstrap_average_precision_ci_low"],
        "bootstrap_average_precision_ci_high": bootstrap["bootstrap_average_precision_ci_high"],
        "brier_score": float(brier_score_loss(y, oof_probs)),
        "expected_calibration_error": float(cal_sum.get("expected_calibration_error", 0.0)),
        "validation_mode": "spatial_group_cv_stacked",
        "top_features": importances[:10],
        "model_card_embedded": model_card_md
    })
    metrics.update(temporal_metrics)
    metrics.update(single_feature_leakage_scan(df))

    return ModelRun(
        model_name=MODEL_NAME,
        description=MODEL_DESCRIPTION,
        model=ensemble,
        predictions=predictions,
        metrics=metrics,
        calibration=cal_sum,
        validation_predictions=validation_predictions,
        coefficient_summary=_format_importance(importances[:5]),
    )


def _make_calibrated_stack() -> StackingClassifier:
    estimators = [
        ("hgb", make_pipeline(StandardScaler(), HistGradientBoostingClassifier(max_iter=150, random_state=42))),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)),
    ]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=1.0),
        cv=3,
        n_jobs=-1,
    )


def _temporal_holdout_metrics(df: pl.DataFrame) -> dict[str, Any]:

    if len(df) < 20:
        return {"temporal_holdout_status": "not_evaluated", "temporal_holdout_reason": "insufficient_rows"}

    years = df["max_resolved_year_train"].fill_null(0.0).to_numpy()
    cutoff = float(np.quantile(years, 0.70))
    train_mask = years < cutoff
    val_mask = years >= cutoff

    if not np.any(train_mask) or not np.any(val_mask):
        return {"temporal_holdout_status": "not_evaluated", "temporal_holdout_reason": "empty_temporal_split"}

    y = df["label_geo_spread"].to_numpy()
    if len(np.unique(y[train_mask])) < 2 or len(np.unique(y[val_mask])) < 2:
        return {"temporal_holdout_status": "not_evaluated", "temporal_holdout_reason": "insufficient_class_diversity"}

    X = _feature_matrix(df, FEATURE_COLUMNS)
    temporal_model = _make_calibrated_stack()
    temporal_model.fit(X[train_mask], y[train_mask])
    probs = temporal_model.predict_proba(X[val_mask])[:, 1]

    val_df = df.filter(val_mask).to_dicts()
    temporal_predictions = [_prediction_from_row(val_df[i], float(probs[i]), MODEL_NAME) for i in range(len(val_df))]

    metrics = evaluate_predictions(temporal_predictions)
    calibration = calibration_summary(temporal_predictions)
    return {
        "temporal_holdout_status": "evaluated",
        "temporal_holdout_cutoff_year": cutoff,
        "temporal_holdout_roc_auc": metrics["roc_auc"],
        "temporal_holdout_average_precision": metrics["average_precision"],
        "temporal_holdout_expected_calibration_error": calibration["expected_calibration_error"],
        "temporal_holdout_brier_score": calibration["brier_score"],
        "temporal_holdout_n_backbones": metrics["n_backbones"],
        "temporal_holdout_n_positive": metrics["n_positive"],
        "temporal_holdout_prevalence": metrics["prevalence"],
    }


def _calculate_permutation_importance(
    model: Any,
    X: NDArray[np.float32],
    y: NDArray[np.integer[Any]],
    columns: tuple[str, ...],
) -> list[dict[str, Any]]:
    baseline_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    importances = []
    rng = np.random.default_rng(7)
    for i, col in enumerate(columns):
        X_shuff = X.copy()
        # Local seeded permutation keeps explanation artifacts byte-stable while
        # avoiding global RNG state pollution across bootstrap/model routines.
        X_shuff[:, i] = X_shuff[rng.permutation(X_shuff.shape[0]), i]
        shuff_auc = roc_auc_score(y, model.predict_proba(X_shuff)[:, 1])
        importances.append({"feature": col, "score": max(0.0, float(baseline_auc - shuff_auc))})
    return sorted(importances, key=lambda x: x["score"], reverse=True)


def _format_importance(importances: list[dict[str, Any]]) -> str:
    return "; ".join([f"{i['feature']}={i['score']:.3f}" for i in importances])


def _feature_matrix(df: pl.DataFrame, columns: tuple[str, ...]) -> NDArray[np.float32]:
    return df.select(columns).fill_null(0.0).cast(pl.Float32).to_numpy()


def _prediction_from_row(row: dict[str, Any], prob: float, model_name: str) -> Prediction:
    knownness = float(row.get("knownness_score", 0.0))
    tier = "review" if (knownness < 0.25 or 0.4 <= prob <= 0.6) else ("high" if prob >= 0.75 or prob <= 0.25 else "medium")
    return Prediction(
        model_name=model_name,
        backbone_id=str(row["backbone_id"]),
        risk_probability=prob,
        confidence_tier=tier,
        label_geo_spread=int(row["label_geo_spread"]),
        knownness_score=knownness,
        n_new_countries_future=int(row["n_new_countries_future"]),
        explanation=f"prob={prob:.2f}; knownness={knownness:.2f}"
    )


def leakage_audit(feature_columns: tuple[str, ...] | list[str]) -> dict[str, Any]:
    blocked = [c for c in feature_columns if any(t in c.lower() for t in LEAKAGE_BLOCKLIST)]
    return {
        "status": "pass" if not blocked else "fail",
        "feature_count": len(feature_columns),
        "blocked_columns": blocked,
        "blocked_tokens": list(LEAKAGE_BLOCKLIST),
    }


def single_feature_leakage_scan(
    rows_or_df: pl.DataFrame | list[GeoSpreadFeatureRow],
    threshold: float = 0.95,
    auc_threshold: float | None = None,
) -> dict[str, float]:
    effective_threshold = float(auc_threshold if auc_threshold is not None else threshold)
    if isinstance(rows_or_df, pl.DataFrame):
        df = rows_or_df
    else:
        payload: list[dict[str, Any]] = []
        for row in rows_or_df:
            item = {
                "backbone_id": row.backbone_id,
                "label_geo_spread": row.label_geo_spread,
                "n_new_countries_future": row.n_new_countries_future,
                "knownness_score": row.knownness_score,
                "max_resolved_year_train": row.max_resolved_year_train,
            }
            item.update(row.features)
            payload.append(item)
        df = pl.DataFrame(payload)

    matrix = _feature_matrix(df, FEATURE_COLUMNS)
    labels = df["label_geo_spread"].to_numpy()
    if len(np.unique(labels)) < 2:
        return {"max_single_feature_auc": 0.5, "suspicious_feature_count": 0.0}
    max_auc = 0.5
    suspicious = 0
    for i in range(matrix.shape[1]):
        auc = roc_auc_score(labels, matrix[:, i])
        auc = max(auc, 1.0 - auc)
        max_auc = max(max_auc, auc)
        if auc >= effective_threshold:
            suspicious += 1
    return {"max_single_feature_auc": max_auc, "suspicious_feature_count": float(suspicious)}


def _feature_summary(model: GeoBioReliabilityModel) -> str:
    gb = next((e for _, e in model.estimators if hasattr(e, "feature_importances_")), None)
    if gb is None:
        return "N/A"
    imps = gb.feature_importances_
    ordered = sorted(zip(FEATURE_COLUMNS, imps), key=lambda x: x[1], reverse=True)
    return "; ".join([f"{n}={v:.3f}" for n, v in ordered[:5]])
