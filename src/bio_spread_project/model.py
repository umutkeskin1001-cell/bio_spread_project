from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from bio_spread_project.config_loader import ModelSpec, load_project_config
from bio_spread_project.features import feature_rows_to_frame
from bio_spread_project.metrics import calibration_summary, evaluate_predictions


@dataclass(frozen=True)
class Prediction:
    model_name: str
    backbone_id: str
    risk_probability: float
    confidence_tier: str
    label_geo_spread: int
    knownness_score: float
    n_new_countries_future: int
    explanation: str
    meta: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        return None


@dataclass(frozen=True)
class ModelRun:
    model_name: str
    description: str
    model: BioSpreadRiskModel
    predictions: list[Prediction]
    metrics: dict[str, Any]
    calibration: dict[str, Any]
    validation_predictions: list[Prediction]
    coefficient_summary: str


class BioSpreadRiskModel:
    def __init__(
        self,
        model_name: str,
        description: str,
        feature_names: tuple[str, ...],
        scaler: StandardScaler,
        clf: LogisticRegression,
    ):
        self.model_name = model_name
        self.description = description
        self.feature_names = feature_names
        self.scaler = scaler
        self.clf = clf

    @classmethod
    def train(cls, df: pl.DataFrame, spec: ModelSpec | None = None) -> BioSpreadRiskModel:
        if isinstance(df, list):
            df = feature_rows_to_frame(df)
        if df.is_empty():
            raise ValueError("Cannot train model without feature rows")
        if spec is None:
            spec = load_project_config().models[0]

        feature_names = tuple(spec.weights.keys())
        X = _extract_matrix(df, feature_names)
        y = df["label_geo_spread"].to_numpy()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = LogisticRegression(C=0.5, class_weight="balanced", random_state=7)
        clf.fit(X_scaled, y)

        return cls(
            model_name=spec.name,
            description=spec.description,
            feature_names=feature_names,
            scaler=scaler,
            clf=clf,
        )

    def predict(self, df: pl.DataFrame) -> list[Prediction]:
        if isinstance(df, list):
            df = feature_rows_to_frame(df)
        X = _extract_matrix(df, self.feature_names)
        X_scaled = self.scaler.transform(X)
        probs = self.clf.predict_proba(X_scaled)[:, 1]

        predictions = []
        # Convert to dicts only for final prediction objects to keep interface compatible
        # though ideally even this should be a DataFrame.
        rows = df.to_dicts()
        for row, prob in zip(rows, probs):
            tier = self._confidence_tier(float(prob), float(row["knownness_score"]))
            predictions.append(
                Prediction(
                    model_name=self.model_name,
                    backbone_id=str(row["backbone_id"]),
                    risk_probability=float(prob),
                    confidence_tier=tier,
                    label_geo_spread=int(row["label_geo_spread"]),
                    knownness_score=float(row["knownness_score"]),
                    n_new_countries_future=int(row["n_new_countries_future"]),
                    explanation=f"prob={prob:.2f}; knownness={row['knownness_score']:.2f}"
                )
            )
        return sorted(predictions, key=lambda p: (-p.risk_probability, p.backbone_id))

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> BioSpreadRiskModel:
        return cast(BioSpreadRiskModel, joblib.load(path))

    @staticmethod
    def _confidence_tier(probability: float, knownness: float) -> str:
        if knownness < 0.25 or 0.40 <= probability <= 0.60:
            return "review"
        if knownness >= 0.55 and (probability >= 0.75 or probability <= 0.25):
            return "high"
        return "medium"


def _extract_matrix(df: pl.DataFrame, names: tuple[str, ...]) -> NDArray[np.float64]:
    exprs = []
    for name in names:
        if name == "mobility":
            exprs.append(pl.col("mean_mobility_score_pre"))
        elif name == "amr":
            exprs.append(pl.col("mean_amr_gene_count_pre").clip(lower_bound=0.0).log1p())
        elif name == "country":
            exprs.append(pl.col("n_countries_pre").cast(pl.Float64).clip(lower_bound=0.0).log1p())
        elif name == "host":
            exprs.append(pl.col("host_diversity_pre").cast(pl.Float64).clip(lower_bound=0.0).log1p())
        elif name == "clinical":
            exprs.append(pl.col("clinical_fraction_pre"))
        elif name == "low_knownness":
            exprs.append(1.0 - pl.col("knownness_score"))
        else:
            raise ValueError(f"Unknown feature requested by model: {name}")

    matrix = df.select(exprs).cast(pl.Float64).to_numpy()
    if not np.isfinite(matrix).all():
        raise ValueError("Model feature matrix contains non-finite values")
    return np.ascontiguousarray(matrix, dtype=np.float64)


def fit_model_surface(df: pl.DataFrame, specs: tuple[ModelSpec, ...]) -> dict[str, ModelRun]:

    if isinstance(df, list):
        df = feature_rows_to_frame(df)
    surface: dict[str, ModelRun] = {}
    for spec in specs:
        model = BioSpreadRiskModel.train(df, spec)
        predictions = model.predict(df)

        feature_names = tuple(spec.weights.keys())
        X = _extract_matrix(df, feature_names)
        y = df["label_geo_spread"].to_numpy()

        oof_probs = np.zeros(len(df))
        class_counts = np.bincount(y.astype(int), minlength=2)
        positive_class_counts = class_counts[class_counts > 0]
        min_class_count = int(positive_class_counts.min()) if len(positive_class_counts) else 0
        n_splits = min(3, min_class_count)
        validation_mode = "cross_validated"

        if n_splits >= 2 and len(np.unique(y)) >= 2:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7)
            for train_idx, val_idx in skf.split(X, y):
                fold_scaler = StandardScaler()
                X_train_scaled = fold_scaler.fit_transform(X[train_idx])
                fold_clf = LogisticRegression(C=0.5, class_weight="balanced", random_state=7)
                fold_clf.fit(X_train_scaled, y[train_idx])
                oof_probs[val_idx] = fold_clf.predict_proba(fold_scaler.transform(X[val_idx]))[:, 1]
        else:
            oof_probs = np.asarray([prediction.risk_probability for prediction in predictions], dtype=float)
            validation_mode = "direct_small_sample"

        validation_predictions = []
        rows = df.to_dicts()
        for i, prob in enumerate(oof_probs):
            row = rows[i]
            validation_predictions.append(
                Prediction(
                    model_name=spec.name,
                    backbone_id=str(row["backbone_id"]),
                    risk_probability=float(prob),
                    confidence_tier="N/A",
                    label_geo_spread=int(row["label_geo_spread"]),
                    knownness_score=float(row["knownness_score"]),
                    n_new_countries_future=int(row["n_new_countries_future"]),
                    explanation=""
                )
            )

        metrics: dict[str, Any] = dict(evaluate_predictions(validation_predictions))
        calibration = calibration_summary(validation_predictions)

        metrics.update({
            "validation_mode": validation_mode,
            "oof_roc_auc": metrics["roc_auc"],
            "oof_average_precision": metrics["average_precision"],
        })

        surface[spec.name] = ModelRun(
            model_name=spec.name,
            description=spec.description,
            model=model,
            predictions=predictions,
            metrics=metrics,
            calibration=calibration,
            validation_predictions=validation_predictions,
            coefficient_summary=_coef_summary(model)
        )
    return surface


def _coef_summary(model: BioSpreadRiskModel) -> str:
    coefs = model.clf.coef_[0]
    ordered = sorted(zip(model.feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)
    return "; ".join([f"{n}={v:.3f}" for n, v in ordered])


def select_primary_model(surface: dict[str, ModelRun]) -> tuple[str, list[dict[str, Any]]]:
    if not surface:
        return "", []

    rows: list[dict[str, Any]] = []
    for name, run in surface.items():
        score = (
            0.40 * run.metrics["roc_auc"] +
            0.30 * run.metrics["average_precision"] +
            0.30 * (1.0 - run.calibration["expected_calibration_error"])
        )
        rows.append({
            "model_name": name,
            "description": run.description,
            "roc_auc": run.metrics["roc_auc"],
            "average_precision": run.metrics["average_precision"],
            "expected_calibration_error": run.calibration["expected_calibration_error"],
            "validation_mode": run.metrics.get("validation_mode", "unknown"),
            "selection_score": score,
            "primary_eligible": True,
            "coefficient_summary": run.coefficient_summary
        })

    rows.sort(key=lambda x: float(x["selection_score"]), reverse=True)
    for i, r in enumerate(rows, 1):
        r["selection_rank"] = float(i)

    return str(rows[0]["model_name"]), rows
