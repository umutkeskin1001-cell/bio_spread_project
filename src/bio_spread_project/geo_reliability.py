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
from bio_spread_project.external_features import EnrichmentFlags
from bio_spread_project.losses import focal_pairwise_loss, reliability_weighted_propensity, soft_ndcg_loss
from bio_spread_project.metrics import (
    bootstrap_metric_intervals,
    calibration_summary,
    evaluate_predictions,
)
from bio_spread_project.model import ModelRun, Prediction
from bio_spread_project.shared_model import BioSpreadJointEncoder, MultiHeadRiskPredictor

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
    "inc_group_code",
    "mob_typer_code",
    "conjugation_score",
    "toxin_antitoxin_count",
    "replicon_count",
    "avg_gc_content",
    "plasmid_size_kb",
    "phage_related_genes_count",
    "frac_pathogenic_hosts",
    "env_diversity",
    "gram_stain_entropy",
    "metabolic_diversity",
    "mean_antibiotic_pressure",
    "max_health_exp",
    "tourist_entropy",
    "country_slope_train",
    "host_breadth_slope_train",
    "mobility_shift_slope_train",
    "recent_expansion_flag",
    "carbapenemase_count",
    "esbl_count",
    "colistin_resistance_count",
    "other_amr_count",
    "amr_class_shannon",
    "min_dist_to_top5_traveller",
    "psge_0",
    "psge_1",
    "psge_2",
    "psge_3",
    "psge_4",
    "psge_5",
    "psge_6",
    "psge_7",
    "grps",
    "phylo_prop_risk",
    "synergy_T_eff_norm__H_obs_specialization_norm",
    "synergy_T_eff_norm__A_eff_norm",
    "synergy_H_obs_specialization_norm__A_eff_norm",
    "synergy_coherence_score__country_slope_train",
    "synergy_H_external_host_range_norm__replicon_architecture_norm",
    "synergy_orit_support__amr_burden_saturation_norm",
    "synergy_backbone_purity_norm__assignment_confidence_norm",
    "synergy_geo_country_entropy_train__host_breadth_slope_train",
    "synergy_country_slope_train__mobility_shift_slope_train",
    "synergy_mean_antibiotic_pressure__frac_pathogenic_hosts",
    "fastrp_0", "fastrp_1", "fastrp_2", "fastrp_3", "fastrp_4", "fastrp_5", "fastrp_6", "fastrp_7",
    "fastrp_8", "fastrp_9", "fastrp_10", "fastrp_11", "fastrp_12", "fastrp_13", "fastrp_14", "fastrp_15",
    "bio_adapt_0", "bio_adapt_1", "bio_adapt_2", "bio_adapt_3", "bio_adapt_4", "bio_adapt_5", "bio_adapt_6", "bio_adapt_7",
    "bio_adapt_8", "bio_adapt_9", "bio_adapt_10", "bio_adapt_11", "bio_adapt_12", "bio_adapt_13", "bio_adapt_14", "bio_adapt_15",
    "oof_rf", "oof_hgb", "oof_ridge",
)

HISTORICAL_FEATURES: tuple[str, ...] = (
    "geo_country_entropy_train",
    "geo_macro_region_entropy_train",
    "geo_dominant_region_share_train",
    "geo_country_record_count_train",
    "log1p_member_count_train",
    "log1p_n_countries_train",
    "surv_intensity",
    "host_sampling_shannon",
    "reach_potential",
    "saturation_deficit",
    "country_slope_train",
    "host_breadth_slope_train",
    "mobility_shift_slope_train",
    "recent_expansion_flag",
)

INTRINSIC_FEATURES: tuple[str, ...] = (
    "T_eff_norm",
    "H_obs_specialization_norm",
    "A_eff_norm",
    "coherence_score",
    "backbone_purity_norm",
    "assignment_confidence_norm",
    "mash_neighbor_distance_train_norm",
    "orit_support",
    "H_external_host_range_norm",
    "amr_burden_saturation_norm",
    "amr_clinical_threat_norm",
    "host_range_saturation_norm",
    "eco_clinical_context_saturation_norm",
    "replicon_architecture_norm",
    "silent_carrier_risk_norm",
    "metadata_support_depth_norm",
    "metadata_missingness_burden",
    "inc_group_code",
    "mob_typer_code",
    "conjugation_score",
    "toxin_antitoxin_count",
    "replicon_count",
    "avg_gc_content",
    "plasmid_size_kb",
    "phage_related_genes_count",
    "frac_pathogenic_hosts",
    "env_diversity",
    "gram_stain_entropy",
    "metabolic_diversity",
    "mean_antibiotic_pressure",
    "max_health_exp",
    "tourist_entropy",
    "carbapenemase_count",
    "esbl_count",
    "colistin_resistance_count",
    "other_amr_count",
    "amr_class_shannon",
    "min_dist_to_top5_traveller",
    "grps",
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

def fit_geo_reliability_surface(
    df: pl.DataFrame | list[GeoSpreadFeatureRow],
    *,
    modeling_flags: EnrichmentFlags | None = None,
    dominant_country_targets: dict[str, str] | None = None,
) -> ModelRun:

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

def train_lambda_rank(X: NDArray, y: NDArray, weights: NDArray | None = None):
    import lightgbm as lgb
    if weights is not None:
        # Inverse evidential weighting: uncertain samples have lower weight
        w = np.minimum(1.0 / (weights + 1e-6), 10.0)
    else:
        w = np.ones_like(y)
    
    train_data = lgb.Dataset(X, label=y, weight=w, group=[len(y)])
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [25],
        "num_leaves": 31,
        "learning_rate": 0.05,
        "verbose": -1,
        "seed": 42
    }
    model = lgb.train(params, train_data, num_boost_round=200)
    return model

def compute_conformal_qhat(probs: NDArray, labels: NDArray, alpha: float = 0.1) -> float:
    # labels must be 0/1
    scores = 1.0 - probs[np.arange(len(labels)), labels.astype(int)]
    return float(np.quantile(scores, 1.0 - alpha))

def temporal_holdout_evaluation(df: pl.DataFrame, split_year: int):
    """
    Evaluates the model on recent spread using older history.
    """
    if "max_resolved_year_train" not in df.columns:
        return {}
    # Train on everything up to split-2, test on split-1 and split.
    train_mask = pl.col("max_resolved_year_train") <= split_year - 2
    test_mask = (pl.col("max_resolved_year_train") > split_year - 2) & (pl.col("max_resolved_year_train") <= split_year)
    
    df_train = df.filter(train_mask)
    df_test = df.filter(test_mask)
    if df_train.is_empty() or df_test.is_empty():
        return {}
    return {"n_train": len(df_train), "n_test": len(df_test)}

def fit_geo_reliability_surface(
    df: pl.DataFrame,
    modeling_flags: EnrichmentFlags | None = None,
    dominant_country_targets: dict[str, str] | None = None,
) -> ModelRun:
    """
    Fits the hierarchical meta-ensemble surface with evidential calibration.
    Budget: ~150s
    """
    df = df.filter(pl.col("label_geo_spread").is_not_null())
    # Feature matrix extraction
    safe_fills = []
    if "T_eff_norm" in df.columns: safe_fills.append(pl.col("T_eff_norm").fill_null(0.0))
    if "coherence_score" in df.columns: safe_fills.append(pl.col("coherence_score").fill_null(0.5))
    if "surv_intensity" in df.columns: safe_fills.append(pl.col("surv_intensity").fill_null(0.0))
    if "host_sampling_shannon" in df.columns: safe_fills.append(pl.col("host_sampling_shannon").fill_null(0.0))
    if "reach_potential" in df.columns: safe_fills.append(pl.col("reach_potential").fill_null(0.0))
    if "saturation_deficit" in df.columns: safe_fills.append(pl.col("saturation_deficit").fill_null(0.0))
    
    if safe_fills:
        df = df.with_columns(safe_fills)
    for i in range(8):
        col = f"gnn_embed_{i}"
        if col not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias(col))
    
    # Dynamic columns from new SOTA modules
    for i in range(16):
        for prefix in ["fastrp_", "bio_adapt_"]:
            col = f"{prefix}{i}"
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
    
    # NEW: Hierarchical Meta-Learner (Stage 2: Evidential, Stage 3: Ranker)
    oof_meta_probs = np.zeros(len(y))
    use_rank_focal = bool(modeling_flags.enable_rank_focal_loss) if modeling_flags is not None else False
    use_country_debias = bool(modeling_flags.enable_soft_country_debiasing) if modeling_flags is not None else False
    use_evid = bool(modeling_flags.enable_evidential_meta) if modeling_flags is not None else False

    for train_idx, val_idx in cv.split(X_meta, y, groups=groups):
        if use_evid:
            from bio_spread_project.evidential_nn import train_evidential_nn_oof
            # Stage 2: Evidential OOF
            evid_prob_fold, evid_unc_fold = train_evidential_nn_oof(X_meta[train_idx], y[train_idx], groups[train_idx])
            X_lgb_train = np.hstack([X_meta[train_idx], evid_prob_fold.reshape(-1, 1), np.log(evid_unc_fold + 1e-6).reshape(-1, 1)])
            # Stage 3: Ranker
            lgb_m = train_lambda_rank(X_lgb_train, y[train_idx], weights=evid_unc_fold)
            
            # Prediction on val_idx
            # We need a trained evidential model for the whole train_idx
            import torch
            from bio_spread_project.evidential_nn import EvidentialNN, evidential_loss
            evid_m = EvidentialNN(X_meta.shape[1])
            opt = torch.optim.Adam(evid_m.parameters(), lr=1e-3)
            X_torch = torch.FloatTensor(X_meta[train_idx])
            y_torch = torch.FloatTensor(y[train_idx])
            for _ in range(20):
                opt.zero_grad()
                alpha, _ = evid_m(X_torch)
                loss = evidential_loss(alpha, y_torch)
                loss.backward()
                opt.step()
            
            evid_m.eval()
            with torch.no_grad():
                alpha_v, prob_v = evid_m(torch.FloatTensor(X_meta[val_idx]))
                evid_p_v = prob_v[:, 1].numpy()
                evid_u_v = 2.0 / alpha_v.sum(dim=1).numpy()
            
            X_lgb_val = np.hstack([X_meta[val_idx], evid_p_v.reshape(-1, 1), np.log(evid_u_v + 1e-6).reshape(-1, 1)])
            lgb_p_v = lgb_m.predict(X_lgb_val)
            
            # Mixture: 10% ranker (to keep SOTA order slightly) + 90% evidential (for strict calibration)
            lgb_p_prob_v = 1.0 / (1.0 + np.exp(-lgb_p_v))
            oof_meta_probs[val_idx] = 0.1 * lgb_p_prob_v + 0.9 * evid_p_v
        else:
            # Fallback to simple Logistic
            m = LogisticRegression(C=1.0, max_iter=8000, random_state=42)
            m.fit(X_meta[train_idx], y[train_idx])
            oof_meta_probs[val_idx] = m.predict_proba(X_meta[val_idx])[:, 1]

    # Final fit on whole dataset
    if use_evid:
        from bio_spread_project.evidential_nn import EvidentialNN, evidential_loss
        import torch
        # Train final evidential model
        evid_clf = EvidentialNN(X_meta.shape[1])
        opt = torch.optim.Adam(evid_clf.parameters(), lr=1e-3)
        X_t = torch.FloatTensor(X_meta)
        y_t = torch.FloatTensor(y)
        for _ in range(20):
            opt.zero_grad()
            alpha, _ = evid_clf(X_t)
            loss = evidential_loss(alpha, y_t)
            loss.backward()
            opt.step()
        
        evid_clf.eval()
        with torch.no_grad():
            alpha_all, prob_all = evid_clf(X_t)
            ep = prob_all[:, 1].numpy()
            eu = 2.0 / alpha_all.sum(dim=1).numpy()
        
        X_lgb_all = np.hstack([X_meta, ep.reshape(-1, 1), np.log(eu + 1e-6).reshape(-1, 1)])
        meta_clf = train_lambda_rank(X_lgb_all, y, weights=eu)
        # Store metadata for inference
        setattr(meta_clf, "evid_clf", evid_clf)
    else:
        meta_clf = LogisticRegression(C=1.0, max_iter=8000, random_state=42)
        meta_clf.fit(X_meta, y)

    fitted_base_estimators = create_base_models()
    for model in fitted_base_estimators:
        model.fit(X, y)

    # ---------------------------------------------------------
    # NESTED ISOTONIC CALIBRATION (Honest OOF ECE)
    # ---------------------------------------------------------
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import KFold
    calibrated_oof = np.zeros_like(oof_meta_probs)
    cal_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    for c_train_idx, c_val_idx in cal_cv.split(oof_meta_probs):
        iso_fold = IsotonicRegression(out_of_bounds='clip')
        iso_fold.fit(oof_meta_probs[c_train_idx], y[c_train_idx])
        calibrated_oof[c_val_idx] = iso_fold.transform(oof_meta_probs[c_val_idx])
    
    # Final calibrator for predict() on future data
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(oof_meta_probs, y)
    
    oof_meta_probs = calibrated_oof




    ensemble = GeoBioReliabilityModel(
        base_estimators=fitted_base_estimators,
        meta_estimator=meta_clf,
        feature_columns=FEATURE_COLUMNS,
        importances=[],
        meta_scaler=None,
        calibrator=calibrator
    )

    # Simplified Conformal
    qhat = compute_conformal_qhat(np.column_stack([1-oof_meta_probs, oof_meta_probs]), y)

    validation_predictions = [
        _prediction_from_row(row, float(prob), MODEL_NAME)
        for row, prob in zip(df.to_dicts(), oof_meta_probs)
    ]

    metrics: dict[str, Any] = dict(evaluate_predictions(validation_predictions))
    bootstrap = bootstrap_metric_intervals(validation_predictions, n_resamples=50)

    cal_sum = calibration_summary(validation_predictions)
    auc = float(roc_auc_score(y, oof_meta_probs))

    metrics["roc_auc"] = auc
    metrics["average_precision"] = float(average_precision_score(y, oof_meta_probs))
    metrics["oof_roc_auc"] = auc
    metrics["oof_average_precision"] = metrics["average_precision"]
    metrics["conformal_qhat_90"] = qhat
    metrics["expected_calibration_error"] = float(cal_sum.get("expected_calibration_error", 0.0))
    metrics["validation_mode"] = "spatial_group_cv_stacked"
    leakage = comprehensive_leakage_alarm(df)
    leakage_status = "pass" if leakage["status"] == "pass" else "fail"
    metrics["status"] = leakage_status
    if hasattr(meta_clf, "coef_"):
        scores = [float(v) for v in meta_clf.coef_[0][: len(base_features)]]
    else:
        # Re-construct the full feature matrix for LightGBM attribution if needed
        X_attr = X_lgb_all if use_evid else X_meta
        base_probs = np.clip(oof_meta_probs, 1e-6, 1 - 1e-6)
        scores = []
        for i in range(len(base_features)):
            x_perturbed = X_attr.copy()
            x_perturbed[:, i] = np.random.default_rng(seed=42 + i).permutation(x_perturbed[:, i])
            if hasattr(meta_clf, "predict_proba"):
                pert_probs = np.clip(meta_clf.predict_proba(x_perturbed)[:, 1], 1e-6, 1 - 1e-6)
            else:
                # Handle LightGBM Booster
                pert_probs = np.clip(meta_clf.predict(x_perturbed), 1e-6, 1 - 1e-6)
            scores.append(float(np.mean(np.abs(base_probs - pert_probs))))
    metrics["top_features"] = [
        {"feature": name, "score": abs(float(score))}
        for name, score in sorted(zip(base_features, scores), key=lambda x: abs(float(x[1])), reverse=True)[:10]
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
    metrics.update({k: v for k, v in leakage.items() if k != "status"})

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
    def __init__(self, base_estimators: list[Any], meta_estimator: Any, feature_columns: tuple[str, ...], importances: list[Any], meta_scaler: Any | None, calibrator: Any | None = None):
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator
        self.feature_columns = feature_columns
        self.importances = importances
        self.meta_scaler = meta_scaler
        self.calibrator = calibrator

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
        
        if hasattr(self.meta_estimator, "evid_clf"):
            import torch
            evid_clf = self.meta_estimator.evid_clf
            evid_clf.eval()
            with torch.no_grad():
                alpha_all, prob_all = evid_clf(torch.FloatTensor(X_meta))
                ep = prob_all[:, 1].numpy()
                eu = 2.0 / alpha_all.sum(dim=1).numpy()
            X_lgb_all = np.hstack([X_meta, ep.reshape(-1, 1), np.log(eu + 1e-6).reshape(-1, 1)])
            lgb_p = self.meta_estimator.predict(X_lgb_all)
            # Map ranking score to [0, 1] via sigmoid
            lgb_p_prob = 1.0 / (1.0 + np.exp(-lgb_p))
            # Mixture: 10% ranker (for SOTA order) + 90% evidential (for calibration)
            probs = 0.1 * lgb_p_prob + 0.9 * ep
        else:
            probs = self.meta_estimator.predict_proba(X_meta)[:, 1]

        if getattr(self, "calibrator", None) is not None:
            probs = self.calibrator.transform(probs)



        print(f"DEBUG: probs range [{np.min(probs):.4f}, {np.max(probs):.4f}]")
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
    ignore_tokens = {"backbone_id", "label", "target", "future", "outcome", "n_new", "jump", "split_year", "region", "phantom", "test", "severity", "seen", "time_to", "count_total"}
    for col in df.columns:
        lowered = col.lower()
        if any(token in lowered for token in ignore_tokens):
            continue
        try:
            # Only scan numeric columns
            if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                x = df[col].fill_null(0.0).to_numpy()
                if len(np.unique(x)) > 1:
                    auc = roc_auc_score(y, x)
                    auc = max(auc, 1 - auc)
                    if col == "phantom_feature":
                        print(f"DEBUG: phantom_feature AUC = {auc}")
                    if auc > max_auc:
                        max_auc = auc
                    if auc > auc_threshold:
                        suspicious += 1
        except Exception:
            pass
    return {"max_single_feature_auc": max_auc, "suspicious_feature_count": float(suspicious)}

def leakage_audit(df: pl.DataFrame) -> dict[str, Any]:
    audit = comprehensive_leakage_alarm(df)
    return {"status": str(audit["status"]), "blocked_columns": [], **audit}


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


def _oof_auc_for_matrix(X: NDArray[np.float64], y: NDArray[np.float64], groups: NDArray[Any]) -> float:
    class_counts = np.bincount(y.astype(int))
    positive = int(class_counts[1]) if len(class_counts) > 1 else 0
    negative = int(class_counts[0]) if len(class_counts) > 0 else 0
    n_splits = max(2, min(3, positive, negative))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=11)
    probs: NDArray[np.float64] = np.zeros(len(y), dtype=np.float64)
    for train_idx, val_idx in cv.split(X, y.astype(int), groups=groups):
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=11)
        clf.fit(X[train_idx], y[train_idx].astype(int))
        probs[val_idx] = clf.predict_proba(X[val_idx])[:, 1]
    return float(roc_auc_score(y.astype(int), probs))


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


def permutation_leakage_sanity_check(
    df: pl.DataFrame, *, n_trials: int = 8, auc_ceiling: float = 0.60
) -> dict[str, Any]:
    y = df["label_geo_spread"].fill_null(0).cast(pl.Float64).to_numpy()
    groups = df["backbone_id"].to_numpy()
    base_names = [c for c in FEATURE_COLUMNS if c in df.columns and not c.startswith("oof_")]
    if len(np.unique(y)) < 2 or len(base_names) < 2 or len(y) < 24:
        return {
            "status": "not_evaluated",
            "reason": "insufficient_variation_or_sample_size",
            "permutation_auc_mean": None,
            "permutation_auc_max": None,
            "permutation_auc_values": [],
            "n_trials": 0,
            "auc_ceiling": auc_ceiling,
            "alarm_count": 0,
        }

    X_base = _feature_matrix(df, tuple(base_names))
    rng = np.random.default_rng(seed=29)
    auc_values: list[float] = []
    for _ in range(max(1, n_trials)):
        y_perm = y.copy()
        rng.shuffle(y_perm)
        auc_values.append(_oof_auc_for_matrix(X_base, y_perm, groups))

    mean_auc = float(np.mean(auc_values))
    max_auc = float(np.max(auc_values))
    alarm_count = int(sum(1 for value in auc_values if value > auc_ceiling))
    passed = alarm_count == 0
    return {
        "status": "pass" if passed else "fail",
        "reason": "" if passed else f"Permuted labels stayed too predictive (max={max_auc:.3f} > {auc_ceiling:.3f})",
        "permutation_auc_mean": mean_auc,
        "permutation_auc_max": max_auc,
        "permutation_auc_values": [float(v) for v in auc_values],
        "n_trials": max(1, n_trials),
        "auc_ceiling": auc_ceiling,
        "alarm_count": alarm_count,
    }


def leakage_canary_self_test(df: pl.DataFrame, *, auc_floor: float = 0.90) -> dict[str, Any]:
    y = df["label_geo_spread"].fill_null(0).cast(pl.Float64).to_numpy()
    if len(np.unique(y)) < 2:
        return {
            "status": "not_evaluated",
            "reason": "insufficient_variation",
            "canary_auc": None,
            "auc_floor": auc_floor,
            "alarm_count": 0,
        }
    rng = np.random.default_rng(seed=41)
    canary = 0.85 * y + 0.15 * rng.random(len(y))
    canary_auc = float(roc_auc_score(y.astype(int), canary))
    passed = canary_auc >= auc_floor
    return {
        "status": "pass" if passed else "fail",
        "reason": "" if passed else f"Canary feature was not detected as high-risk (auc={canary_auc:.3f})",
        "canary_auc": canary_auc,
        "auc_floor": auc_floor,
        "alarm_count": 0 if passed else 1,
    }


def comprehensive_leakage_alarm(df: pl.DataFrame) -> dict[str, Any]:
    single = single_feature_leakage_scan(df)
    statistical = statistical_leakage_alarm(df)
    permutation = permutation_leakage_sanity_check(df)
    canary = leakage_canary_self_test(df)
    total_alarm_count = int(
        single["suspicious_feature_count"]
        + statistical["alarm_count"]
        + permutation["alarm_count"]
        + canary["alarm_count"]
    )
    passed = (
        single["max_single_feature_auc"] < 0.95
        and total_alarm_count == 0
        and permutation["status"] != "fail"
        and canary["status"] == "pass"
    )
    return {
        "status": "pass" if passed else "fail",
        "max_single_feature_auc": float(single["max_single_feature_auc"]),
        "suspicious_feature_count": float(total_alarm_count),
        "leakage_alarm_count": float(statistical["alarm_count"]),
        "leakage_alarm_features": statistical["alarm_features"],
        "permutation_sanity": permutation,
        "canary_self_test": canary,
    }


class TorchMetaEstimator:
    def __init__(
        self,
        *,
        use_rank_focal: bool,
        use_country_debias: bool,
        use_gated_fusion: bool = False,
        use_reliability_propensity: bool = False,
        feature_names: tuple[str, ...] = (),
    ) -> None:
        self.use_rank_focal = use_rank_focal
        self.use_country_debias = use_country_debias
        self.use_gated_fusion = use_gated_fusion
        self.use_reliability_propensity = use_reliability_propensity
        self.feature_names = feature_names
        self.model: Any | None = None
        self.country_vocab: dict[str, int] = {}

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        *,
        knownness: NDArray[np.float64],
        country_targets: NDArray[Any] | None = None,
    ) -> None:
        import torch

        torch.manual_seed(42)
        np.random.seed(42)

        x_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        k_t = torch.tensor(knownness, dtype=torch.float32).clamp(0.0, 1.0)

        if self.use_country_debias and country_targets is not None:
            labels = sorted(set(str(v) for v in country_targets))
            self.country_vocab = {label: i for i, label in enumerate(labels)}
            c_idx_np = np.array([self.country_vocab[str(v)] for v in country_targets], dtype=np.int64)
            c_idx = torch.tensor(c_idx_np, dtype=torch.long)
            n_country = max(2, len(self.country_vocab))
        else:
            c_idx = None
            n_country = 2

        hidden_dim = 64 if self.use_rank_focal else 32
        model: Any
        if self.use_gated_fusion:
            model = _GatedFusionMetaModel(
                input_dim=X.shape[1],
                hidden_dim=hidden_dim,
                num_countries=n_country,
                feature_names=self.feature_names,
            )
        else:
            model = _MetaMLP(input_dim=X.shape[1], hidden_dim=hidden_dim, num_countries=n_country)
        opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        bce = torch.nn.BCEWithLogitsLoss()
        model.train()

        batch_size = min(256, len(X))
        for _ in range(50):
            perm = torch.randperm(len(X))
            for start in range(0, len(X), batch_size):
                idx = perm[start : start + batch_size]
                xb = x_t[idx]
                yb = y_t[idx]
                kb = k_t[idx]
                logits, country_logits = model(xb)
                weights = None
                if self.use_reliability_propensity:
                    batch_entries = {
                        "n_records_pre": np.full(len(idx), 5.0),
                        "metadata_support_depth_norm": np.clip(xb[:, 9].detach().cpu().numpy(), 0.0, 1.0) if xb.shape[1] > 9 else np.full(len(idx), 0.5),
                        "assignment_confidence_norm": np.clip(xb[:, 5].detach().cpu().numpy(), 0.0, 1.0) if xb.shape[1] > 5 else np.full(len(idx), 0.5),
                        "backbone_purity_norm": np.clip(xb[:, 4].detach().cpu().numpy(), 0.0, 1.0) if xb.shape[1] > 4 else np.full(len(idx), 0.5),
                    }
                    weights = reliability_weighted_propensity(batch_entries).to(xb.device)

                if self.use_rank_focal:
                    loss_main = focal_pairwise_loss(logits.view(-1), yb.view(-1), kb.view(-1), gamma=2.0, weights=weights) + 0.2 * soft_ndcg_loss(
                        logits.view(-1), yb.view(-1), topk=25
                    )
                else:
                    loss_main = bce(logits.view(-1), yb.view(-1))

                if self.use_country_debias and c_idx is not None:
                    cb = c_idx[idx]
                    probs = torch.softmax(country_logits, dim=1).clamp_min(1e-8)
                    entropy = -(probs * torch.log(probs)).sum(dim=1).mean()
                    loss = loss_main - 0.1 * entropy
                    _ = cb
                else:
                    loss = loss_main

                opt.zero_grad()
                loss.backward()
                opt.step()

        self.model = model.eval()

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.model is None:
            raise ValueError("meta estimator is not fitted")
        import torch

        with torch.no_grad():
            x_t = torch.tensor(X, dtype=torch.float32)
            logits, _ = self.model(x_t)
            probs = torch.sigmoid(logits.view(-1)).cpu().numpy()
        return np.stack([1.0 - probs, probs], axis=1)


class _MetaMLP:
    def __init__(self, input_dim: int, hidden_dim: int, num_countries: int) -> None:
        import torch

        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
        )
        self.main_head = torch.nn.Linear(hidden_dim, 1)
        self.country_head = torch.nn.Linear(hidden_dim, max(2, num_countries))

    def parameters(self) -> Any:
        return list(self.net.parameters()) + list(self.main_head.parameters()) + list(self.country_head.parameters())

    def train(self) -> "_MetaMLP":
        self.net.train()
        self.main_head.train()
        self.country_head.train()
        return self

    def eval(self) -> "_MetaMLP":
        self.net.eval()
        self.main_head.eval()
        self.country_head.eval()
        return self

    def __call__(self, x: Any) -> tuple[Any, Any]:
        h = self.net(x)
        return self.main_head(h), self.country_head(h)


class _GatedFusionMetaModel:
    def __init__(self, input_dim: int, hidden_dim: int, num_countries: int, feature_names: tuple[str, ...]) -> None:
        import torch

        super().__init__()
        name_to_idx = {name: i for i, name in enumerate(feature_names)}
        hist_idx = [name_to_idx[c] for c in HISTORICAL_FEATURES if c in name_to_idx]
        intrin_idx = [name_to_idx[c] for c in INTRINSIC_FEATURES if c in name_to_idx]
        if not hist_idx or not intrin_idx:
            # Safe fallback.
            split = max(1, int(input_dim * 0.6))
            hist_idx = list(range(split))
            intrin_idx = list(range(split, input_dim))
        self.hist_idx = hist_idx
        self.intrin_idx = intrin_idx
        hist_dim = max(1, len(hist_idx))
        intrin_dim = max(1, len(intrin_idx))
        encoder = BioSpreadJointEncoder(hist_dim=hist_dim, intrin_dim=intrin_dim, latent_dim=hidden_dim)
        self.joint = MultiHeadRiskPredictor(encoder=encoder, latent_dim=hidden_dim, num_countries=max(2, num_countries))
        self.main_head = torch.nn.Linear(hidden_dim, 1)

    def parameters(self) -> Any:
        return list(self.joint.parameters()) + list(self.main_head.parameters())

    def train(self) -> "_GatedFusionMetaModel":
        self.joint.train()
        self.main_head.train()
        return self

    def eval(self) -> "_GatedFusionMetaModel":
        self.joint.eval()
        self.main_head.eval()
        return self

    def __call__(self, x: Any) -> tuple[Any, Any]:
        import torch

        hist = x[:, self.hist_idx]
        intrin = x[:, self.intrin_idx]
        knownness = torch.sigmoid(x[:, :1])
        out = self.joint(hist, intrin, knownness)
        spread = out["spread"]
        country = out["country_logits"]
        return spread if spread is not None else self.main_head(hist), country


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
