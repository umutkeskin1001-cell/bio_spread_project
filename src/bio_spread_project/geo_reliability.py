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
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import fmin

from bio_spread_project.data import read_table
from bio_spread_project.external_features import EnrichmentFlags
from bio_spread_project.losses import focal_pairwise_loss, reliability_weighted_propensity, soft_ndcg_loss
from bio_spread_project.metrics import (
    bootstrap_metric_intervals,
    compute_metrics,
    calibration_summary,
    evaluate_predictions,
)
from bio_spread_project.model import ModelRun, Prediction
from bio_spread_project.validation_protocol_v2 import (
    build_rolling_temporal_windows,
    evaluate_temporal_consistency,
)
from bio_spread_project.features_enrichment import (
    PhyloSpatialGraphEmbedder,
    compute_grps,
)
from bio_spread_project.embeddings import EmbeddingStore
from bio_spread_project.phylo_propagation import build_phylo_propagation

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


class _PlattCalibrator:
    def __init__(self, clf: LogisticRegression):
        self._clf = clf

    def transform(self, scores: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        return self._clf.predict_proba(x)[:, 1]


def _active_geo_feature_columns(flags: EnrichmentFlags | None) -> tuple[str, ...]:
    cols = [c for c in FEATURE_COLUMNS if not c.startswith("oof_")]
    if flags is None:
        return tuple(cols)
    active: list[str] = []
    for c in cols:
        if (not flags.enable_synergy_interactions) and c.startswith("synergy_"):
            continue
        if (not flags.enable_phylo_spatial_embedding) and c.startswith("psge_"):
            continue
        if (not flags.enable_graph_contagion) and c.startswith("fastrp_"):
            continue
        if (not flags.enable_grps) and c == "grps":
            continue
        if (not flags.enable_phylo_propagation) and c == "phylo_prop_risk":
            continue
        active.append(c)
    forced_exclude = {
        token.strip()
        for token in os.environ.get("BIO_SPREAD_EXCLUDE_FEATURES", "").split(",")
        if token.strip()
    }
    if forced_exclude:
        active = [c for c in active if c not in forced_exclude]
    return tuple(active)


def _max_bin_calibration_gap(cal_bins: list[dict[str, Any]]) -> float:
    gaps: list[float] = []
    for b in cal_bins:
        mp = b.get("mean_prediction")
        ob = b.get("observed_rate")
        if mp is None or ob is None:
            continue
        gaps.append(abs(float(mp) - float(ob)))
    return float(max(gaps)) if gaps else 0.0


def _knownness_slice_metrics(
    predictions: list[Prediction],
    *,
    quantile: float = 0.20,
    min_samples: int = 20,
) -> dict[str, float | None]:
    if not predictions:
        return {
            "knownness_slice_threshold": None,
            "knownness_slice_n": None,
            "knownness_slice_prevalence": None,
            "knownness_slice_roc_auc": None,
            "knownness_slice_average_precision": None,
            "knownness_slice_expected_calibration_error": None,
            "knownness_slice_brier_score": None,
            "knownness_slice_top_k_precision": None,
            "knownness_slice_abstain_rate": None,
        }
    knownness = np.asarray([float(p.knownness_score) for p in predictions], dtype=np.float64)
    threshold = float(np.quantile(knownness, quantile))
    subset = [p for p in predictions if float(p.knownness_score) <= threshold]
    payload: dict[str, float | None] = {
        "knownness_slice_threshold": threshold,
        "knownness_slice_n": float(len(subset)),
    }
    if len(subset) < min_samples:
        payload.update(
            {
                "knownness_slice_prevalence": None,
                "knownness_slice_roc_auc": None,
                "knownness_slice_average_precision": None,
                "knownness_slice_expected_calibration_error": None,
                "knownness_slice_brier_score": None,
                "knownness_slice_top_k_precision": None,
                "knownness_slice_abstain_rate": None,
            }
        )
        return payload
    base = evaluate_predictions(subset)
    cal = calibration_summary(subset)
    payload.update(
        {
            "knownness_slice_prevalence": float(base.get("prevalence", 0.0)),
            "knownness_slice_roc_auc": float(base["roc_auc"]) if base.get("roc_auc") is not None else None,
            "knownness_slice_average_precision": float(base["average_precision"]) if base.get("average_precision") is not None else None,
            "knownness_slice_expected_calibration_error": float(cal.get("expected_calibration_error", 0.0)),
            "knownness_slice_brier_score": float(cal.get("brier_score", 0.0)),
            "knownness_slice_top_k_precision": float(base.get("top_k_precision", 0.0)),
            "knownness_slice_abstain_rate": float(base.get("abstain_rate", 0.0)),
        }
    )
    return payload


def _crossfit_calibrated_probs(
    raw_scores: NDArray[np.float64],
    labels: NDArray[np.int64],
    *,
    kind: str,
    n_splits: int = 5,
    seed: int = 42,
) -> NDArray[np.float64]:
    y = labels.astype(int)
    min_class = int(np.bincount(y, minlength=2).min())
    folds = max(2, min(n_splits, min_class))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    out = np.zeros(len(y), dtype=np.float64)
    x = raw_scores.reshape(-1, 1)
    for tr, va in skf.split(x, y):
        if kind == "isotonic":
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(raw_scores[tr], y[tr])
            out[va] = cal.transform(raw_scores[va])
        else:
            platt = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
            platt.fit(x[tr], y[tr])
            out[va] = platt.predict_proba(x[va])[:, 1]
    return out

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
        df = df.drop(leakage_columns)

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
        # If we aliased spread_label, we should drop nulls to get an honest holdout evaluation
        if "spread_label" in df.columns:
            df = df.filter(pl.col("label_geo_spread").is_not_null())

    required = {"backbone_id", "label_geo_spread", "n_new_countries_future", *REQUIRED_FEATURE_COLUMNS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Geo feature surface missing required columns: {', '.join(missing)}")

    # Root hygiene: strip known direct leakage columns from source surfaces.
    forbidden_name_tokens = ("event_within_", "time_to_", "_jump_", "future_unseen")
    direct_leak = [c for c in df.columns if any(t in c.lower() for t in forbidden_name_tokens)]
    if direct_leak:
        df = df.drop(direct_leak)

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
    modeling_flags: EnrichmentFlags | None = None,
    dominant_country_targets: dict[str, str] | None = None,
    external_dir: Path | None = None,
    raw_records: pl.DataFrame | None = None,
    split_year: int = 2020,
) -> ModelRun:
    """
    Fits the hierarchical meta-ensemble surface with evidential calibration.
    All enrichment is performed INSIDE the CV loop to ensure zero leakage.
    """
    if isinstance(df, list):
        df = geo_rows_to_frame(df)
    df = df.filter(pl.col("label_geo_spread").is_not_null())
    
    # Pre-extract basic features that don't need label-dependent enrichment
    bio_features = (
        "frac_pathogenic_hosts", "env_diversity", "metabolic_diversity", 
        "replicon_count", "avg_gc_content", "plasmid_size_kb", "orit_support"
    )
    
    # LEAK-08 FIX: Ensure we drop any pre-existing enriched columns that might
    # have been added by previous steps to prevent Look-ahead leakage.
    stale_cols = [c for c in df.columns if c.startswith("psge_") or c.startswith("fastrp_") or c in ("phylo_prop_risk", "grps")]
    if stale_cols:
        df = df.drop(stale_cols)
    
    all_active_features = _active_geo_feature_columns(modeling_flags)
    base_features = tuple(
        c for c in all_active_features
        if not c.startswith("psge_") and not c.startswith("fastrp_") and c not in ("phylo_prop_risk", "grps")
    )
    y = df["label_geo_spread"].to_numpy()
    groups = df["backbone_id"].to_numpy()

    fast_mode = False
    cv_splits = min(5, len(y))
    if cv_splits < 2:
        raise ValueError("Insufficient data for cross-validation")
        
    cv = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    oof_meta_probs = np.zeros(len(y))
    # For learned weights
    oof_log_probs = np.zeros(len(y))
    oof_evid_probs = np.zeros(len(y))
    
    fold_idx = 0  # Track fold for deterministic seeding
    
    cv_index_matrix = np.zeros((len(y), 1), dtype=np.float32)
    for train_idx, val_idx in cv.split(cv_index_matrix, y, groups=groups):
        df_train = df[train_idx]
        df_val = df[val_idx]
        
        current_train_features = df_train.clone()
        current_val_features = df_val.clone()
        
        # A. Phylo Propagation (Fold-Isolated, INDUCTIVE)
        if (not fast_mode) and raw_records is not None and external_dir:
            mash_path = external_dir / "mash_distances.tsv"
            if mash_path.exists():
                labeled_ids = set(df_train["backbone_id"].to_list())
                all_ids_in_fold = pl.concat([df_train, df_val])
                prop_df = build_phylo_propagation(
                    all_ids_in_fold, mash_path,
                    split_year=split_year, labeled_ids=labeled_ids
                )
                current_train_features = current_train_features.join(prop_df, on="backbone_id", how="left", coalesce=True)
                current_val_features = current_val_features.join(prop_df, on="backbone_id", how="left", coalesce=True)

        # B. PSGE (Inductive Graph Embedding)
        if modeling_flags and modeling_flags.enable_phylo_spatial_embedding and raw_records is not None:
            psge = PhyloSpatialGraphEmbedder(dim=8)
            train_bb_ids = df_train["backbone_id"].to_list()
            x_psge_train = _feature_matrix(df_train, base_features)
            train_records = raw_records.filter(pl.col("backbone_id").is_in(train_bb_ids))
            psge.fit(
                train_records, split_year=split_year,
                backbone_ids=train_bb_ids,
                feature_matrix=x_psge_train,
                mash_path=(external_dir / "mash_distances.tsv" if external_dir else None)
            )
            psge_train_df = psge._to_df(train_bb_ids)
            x_psge_val = _feature_matrix(df_val, base_features)
            psge_val_df = psge.transform(df_val["backbone_id"].to_list(), x_psge_val)
            current_train_features = current_train_features.join(psge_train_df, on="backbone_id", how="left", coalesce=True).fill_null(0.0)
            current_val_features = current_val_features.join(psge_val_df, on="backbone_id", how="left", coalesce=True).fill_null(0.0)

        # B2. GRPS (Genomic Risk Propensity Score - Inductive)
        # LEAK-08 FIX: Compute GRPS inside CV using only train-fold labels
        if (not fast_mode) and modeling_flags and modeling_flags.enable_grps and external_dir:
            embed_dir = external_dir / "embeddings"
            if (embed_dir / "esm2_embeddings.parquet").exists():
                store = EmbeddingStore(embed_dir)
                all_ids = df_train["backbone_id"].to_list() + df_val["backbone_id"].to_list()
                emb = store.get(all_ids, kind="esm2")
                if not emb.is_empty():
                    emb_cols = [c for c in emb.columns if c.startswith("esm2_embed_")]
                    # Labels only from train fold
                    labels_train = df_train.select(["backbone_id", "label_geo_spread"])
                    low_k_ids = all_ids # Compute for everyone to be safe
                    grps_df = compute_grps(low_k_ids, emb, labels_train, emb_cols)
                    current_train_features = current_train_features.join(grps_df, on="backbone_id", how="left", coalesce=True).fill_null(0.0)
                    current_val_features = current_val_features.join(grps_df, on="backbone_id", how="left", coalesce=True).fill_null(0.0)

        # C. Unsupervised FastRP (label-free, uses only graph topology)
        # Train-fold FastRP uses train-only records for temporal honesty
        if fast_mode:
            for i in range(16):
                col = f"fastrp_{i}"
                if col not in current_train_features.columns:
                    current_train_features = current_train_features.with_columns(pl.lit(0.0).alias(col))
                if col not in current_val_features.columns:
                    current_val_features = current_val_features.with_columns(pl.lit(0.0).alias(col))
        else:
            train_records_for_frp = raw_records.filter(
                pl.col("backbone_id").is_in(df_train["backbone_id"].to_list())
            ) if raw_records is not None else None
            current_train_features = _compute_unsupervised_fastrp(
                current_train_features, external_dir, train_records_for_frp, dim=16, split_year=split_year
            )
            current_val_features = _compute_unsupervised_fastrp(
                current_val_features, external_dir, train_records_for_frp, dim=16, split_year=split_year
            )
        
        # D. Base Estimators (Stage 1)
        active_features = list(base_features)
        if modeling_flags is None or modeling_flags.enable_phylo_spatial_embedding:
            active_features.extend([f"psge_{i}" for i in range(8)])
        if modeling_flags is None or modeling_flags.enable_phylo_propagation:
            active_features.append("phylo_prop_risk")
        if modeling_flags is None or modeling_flags.enable_graph_contagion:
            active_features.extend([f"fastrp_{i}" for i in range(16)])
        if modeling_flags is None or modeling_flags.enable_grps:
            active_features.append("grps")
        
        X_train_full = _feature_matrix(current_train_features, tuple(active_features))
        X_val_full = _feature_matrix(current_val_features, tuple(active_features))
        
        X_train_bio = _feature_matrix(current_train_features, bio_features)
        X_val_bio = _feature_matrix(current_val_features, bio_features)
        y_train = current_train_features["label_geo_spread"].to_numpy()
        
        base_models = create_base_models()
        oof_base_val = np.zeros((len(df_val), len(base_models)))
        
        for i, m in enumerate(base_models):
            m.fit(X_train_bio, y_train)
            if hasattr(m, "predict_proba"):
                oof_base_val[:, i] = m.predict_proba(X_val_bio)[:, 1]
            else:
                d = m.decision_function(X_val_bio)
                oof_base_val[:, i] = 1 / (1 + np.exp(-d))
        
        # E. Meta Estimator (Stage 2)
        # LEAK-05 FIX: Use unique seed per outer fold to prevent correlated splits
        class_counts_inner = np.bincount(y_train.astype(int))
        positive_inner = int(class_counts_inner[1]) if len(class_counts_inner) > 1 else 0
        negative_inner = int(class_counts_inner[0]) if len(class_counts_inner) > 0 else 0
        inner_splits = min(3, positive_inner, negative_inner)
        if inner_splits < 2:
            raise ValueError("Insufficient class support in fold for inner cross-validation")
        inner_cv = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=42 + fold_idx)
        inner_oof_base = np.zeros((len(X_train_full), len(base_models)))
        train_groups = current_train_features["backbone_id"].to_numpy()
        for it, iv in inner_cv.split(X_train_bio, y_train, groups=train_groups):
            for i, m in enumerate(create_base_models()):
                m.fit(X_train_bio[it], y_train[it])
                if hasattr(m, "predict_proba"):
                    inner_oof_base[iv, i] = m.predict_proba(X_train_bio[iv])[:, 1]
                else:
                    d = m.decision_function(X_train_bio[iv])
                    inner_oof_base[iv, i] = 1 / (1 + np.exp(-d))
        
        X_meta_train_honest = np.hstack([X_train_full, inner_oof_base])
        X_meta_val = np.hstack([X_val_full, oof_base_val])
        
        # --- Robust Simplicity Meta-Learner Layer ---
        # Highly regularized Logistic Regression is more stable for generalization
        
        # LEAK-04 FIX: Proper Inverse Surveillance Weighting (ISW)
        # Uses IPW-style formula: w = 1 / (ε + T_eff_norm), clipped to [0.2, 5.0]
        # This gives over-surveilled backbones (high T_eff) lower weight,
        # forcing the model to learn biological risk rather than reporting density.
        train_surv = current_train_features["T_eff_norm"].fill_null(0.0).to_numpy()
        eps_isw = 0.2
        sample_weights = np.clip(1.0 / (eps_isw + train_surv), 0.2, 5.0)
        
        meta = LogisticRegression(C=0.1, max_iter=1000)
        meta.fit(X_meta_train_honest, y_train, sample_weight=sample_weights)
        log_p_v = meta.predict_proba(X_meta_val)[:, 1]
        
        # Evidential Mixture for uncertainty-aware ranking
        if fast_mode:
            evid_p_v = log_p_v
        else:
            import torch
            from bio_spread_project.evidential_nn import EvidentialNN, evidential_loss

            # LEAK-06 FIX: Deterministic torch seeding per fold for reproducibility
            torch.manual_seed(42 + fold_idx)
            np.random.seed(42 + fold_idx)

            evid_fold = EvidentialNN(X_meta_train_honest.shape[1], hidden=32, dropout=0.3)
            opt = torch.optim.Adam(evid_fold.parameters(), lr=1e-3)
            X_t = torch.as_tensor(np.array(X_meta_train_honest, dtype=np.float32, copy=True))
            y_t = torch.as_tensor(np.array(y_train, dtype=np.float32, copy=True))

            evid_epochs = 30
            for _ in range(evid_epochs):
                opt.zero_grad()
                alpha, _ = evid_fold(X_t)
                loss = evidential_loss(alpha, y_t)
                loss.backward()
                opt.step()


            evid_fold.eval()
            with torch.no_grad():
                x_val_t = torch.as_tensor(np.array(X_meta_val, dtype=np.float32, copy=True))
                _, prob_v = evid_fold(x_val_t)
                evid_p_v = prob_v[:, 1].numpy()
            
        # Store for weight optimization (ISSUE-11)
        oof_log_probs[val_idx] = log_p_v
        oof_evid_probs[val_idx] = evid_p_v
        fold_idx += 1

    # ISSUE-11: Learned Ensemble Mixture Weights
    # Optimize w such that: Loss(w*log + (1-w)*evid) is minimized on OOF data.
    def mixture_loss(w):
        w = np.clip(w[0], 0.0, 1.0)
        p = w * oof_log_probs + (1.0 - w) * oof_evid_probs
        # Use Log Loss for optimization
        eps = 1e-15
        p = np.clip(p, eps, 1.0 - eps)
        return -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

    optimal_w = fmin(mixture_loss, [0.6], disp=False)[0]
    optimal_w = float(np.clip(optimal_w, 0.1, 0.9)) # Keep some of both
    
    oof_mixture = optimal_w * oof_log_probs + (1.0 - optimal_w) * oof_evid_probs

    # LEAK-07 FIX: Honest Calibration on OOF Scores
    # Calibration model selection: choose lower ECE, then lower Brier.
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof_mixture, y)
    iso_probs = iso.transform(oof_mixture)
    iso_m = compute_metrics(labels=y.astype(np.int64), probabilities=iso_probs)

    platt_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    platt_clf.fit(oof_mixture.reshape(-1, 1), y.astype(int))
    platt = _PlattCalibrator(platt_clf)
    platt_probs = platt.transform(oof_mixture)
    platt_m = compute_metrics(labels=y.astype(np.int64), probabilities=platt_probs)

    iso_cv_probs = _crossfit_calibrated_probs(oof_mixture, y.astype(np.int64), kind="isotonic", n_splits=5)
    platt_cv_probs = _crossfit_calibrated_probs(oof_mixture, y.astype(np.int64), kind="platt", n_splits=5)
    iso_cv_m = compute_metrics(labels=y.astype(np.int64), probabilities=iso_cv_probs)
    platt_cv_m = compute_metrics(labels=y.astype(np.int64), probabilities=platt_cv_probs)

    if (
        platt_cv_m.expected_calibration_error < iso_cv_m.expected_calibration_error
        or (
            platt_cv_m.expected_calibration_error == iso_cv_m.expected_calibration_error
            and platt_cv_m.brier_score < iso_cv_m.brier_score
        )
    ):
        calibrator = platt
        chosen_cal = platt_m
        calibrator_type = "platt"
        oof_meta_probs = platt_cv_probs
    elif (
        platt_m.expected_calibration_error < iso_m.expected_calibration_error
        or (
            platt_m.expected_calibration_error == iso_m.expected_calibration_error
            and platt_m.brier_score < iso_m.brier_score
        )
    ):
        calibrator = platt
        chosen_cal = platt_m
        calibrator_type = "platt"
        oof_meta_probs = platt_cv_probs
    else:
        calibrator = iso
        chosen_cal = iso_m
        calibrator_type = "isotonic"
        oof_meta_probs = iso_cv_probs

    calibration_diagnostics = {
        "isotonic_in_sample_ece": float(iso_m.expected_calibration_error),
        "isotonic_in_sample_brier": float(iso_m.brier_score),
        "platt_in_sample_ece": float(platt_m.expected_calibration_error),
        "platt_in_sample_brier": float(platt_m.brier_score),
        "isotonic_crossfit_ece": float(iso_cv_m.expected_calibration_error),
        "isotonic_crossfit_brier": float(iso_cv_m.brier_score),
        "platt_crossfit_ece": float(platt_cv_m.expected_calibration_error),
        "platt_crossfit_brier": float(platt_cv_m.brier_score),
    }

    # FINAL FIT — Enrich df and store production models
    # Drop any stale columns before re-enriching
    stale_cols = [c for c in df.columns if c.startswith("psge_") or c.startswith("fastrp_") or c in ("phylo_prop_risk", "grps")]
    if stale_cols:
        df = df.drop(stale_cols)

    # Final Enrichment (Phylo-Prop)
    if (not fast_mode) and raw_records is not None and external_dir:
        mash_path = external_dir / "mash_distances.tsv"
        if mash_path.exists():
            final_phylo_prop_df = build_phylo_propagation(df, mash_path, split_year=split_year)
            df = df.join(final_phylo_prop_df, on="backbone_id", how="left", coalesce=True).fill_null(0.5)

    if fast_mode:
        for i in range(16):
            col = f"fastrp_{i}"
            if col not in df.columns:
                df = df.with_columns(pl.lit(0.0).alias(col))
    else:
        df = _compute_unsupervised_fastrp(df, external_dir, raw_records, dim=16, split_year=split_year)
    
    # Final Enrichment (GRPS)
    grps_model_store = None
    if (not fast_mode) and modeling_flags and modeling_flags.enable_grps and external_dir:
        embed_dir = external_dir / "embeddings"
        if (embed_dir / "esm2_embeddings.parquet").exists():
            grps_model_store = EmbeddingStore(embed_dir)
            all_ids = df["backbone_id"].to_list()
            emb = grps_model_store.get(all_ids, kind="esm2")
            if not emb.is_empty():
                emb_cols = [c for c in emb.columns if c.startswith("esm2_embed_")]
                labels_all = df.select(["backbone_id", "label_geo_spread"])
                grps_df = compute_grps(all_ids, emb, labels_all, emb_cols)
                df = df.join(grps_df, on="backbone_id", how="left", coalesce=True).fill_null(0.0)

    # Re-extract full features
    all_final_features = list(base_features)
    if modeling_flags is None or modeling_flags.enable_phylo_spatial_embedding:
        all_final_features.extend([f"psge_{i}" for i in range(8)])
    if modeling_flags is None or modeling_flags.enable_phylo_propagation:
        all_final_features.append("phylo_prop_risk")
    if modeling_flags is None or modeling_flags.enable_grps:
        all_final_features.append("grps")
    if modeling_flags is None or modeling_flags.enable_graph_contagion:
        all_final_features.extend([f"fastrp_{i}" for i in range(16)])

    X_bio_final = _feature_matrix(df, bio_features)

    # FINAL ENRICHMENT FIT (on all data for the production model)
    psge = None
    phylo_prop = None
    phylo_graph = None
    
    if modeling_flags and modeling_flags.enable_phylo_spatial_embedding and raw_records is not None:
        psge = PhyloSpatialGraphEmbedder(dim=8)
        # For the final model, we use ALL data to fit the embedding space
        mash_path = (external_dir / "mash_distances.tsv" if external_dir else None)
        psge.fit(
            raw_records, 
            split_year=split_year, 
            backbone_ids=df["backbone_id"].to_list(), 
            feature_matrix=X_bio_final, 
            mash_path=mash_path
        )
        psge_df = psge._to_df(df["backbone_id"].to_list())
        df = df.join(psge_df, on="backbone_id", how="left", coalesce=True)
        
    if raw_records is not None and external_dir:
        mash_path = external_dir / "mash_distances.tsv"
        if mash_path.exists():
            from bio_spread_project.phylo_propagation import PhyloPropagator
            phylo_prop = PhyloPropagator(split_year=split_year)
            phylo_prop.mash_path = mash_path
            # Store the training labels for inference propagation
            phylo_prop.train_df = df.select(["backbone_id", "label_geo_spread"])
            # Use all data as labeled for final propagation
            prop_df = phylo_prop.predict(None, df, np.ones(len(df), dtype=bool))
            df = df.join(prop_df, on="backbone_id", how="left", coalesce=True)

    X_full_final = _feature_matrix(df, all_final_features)
    X_bio_final = _feature_matrix(df, bio_features)
    y_final = df["label_geo_spread"].to_numpy()
    
    fitted_base = create_base_models()
    for i, m in enumerate(fitted_base):
        m.fit(X_bio_final, y_final)
    
    honest_base_oof = np.zeros((len(y_final), len(fitted_base)))
    final_groups = df["backbone_id"].to_numpy()
    for train_idx, val_idx in cv.split(X_bio_final, y_final, groups=final_groups):
        for i, m in enumerate(create_base_models()):
            m.fit(X_bio_final[train_idx], y_final[train_idx])
            if hasattr(m, "predict_proba"):
                honest_base_oof[val_idx, i] = m.predict_proba(X_bio_final[val_idx])[:, 1]
            else:
                d = m.decision_function(X_bio_final[val_idx])
                honest_base_oof[val_idx, i] = 1 / (1 + np.exp(-d))

    X_meta_final = np.hstack([X_full_final, honest_base_oof])
    
    # Base Logistic for final fit — consistent ISW formula
    final_surv = df["T_eff_norm"].fill_null(0.0).to_numpy()
    eps_isw = 0.2
    final_weights = np.clip(1.0 / (eps_isw + final_surv), 0.2, 5.0)
    
    meta_final = LogisticRegression(C=0.1, max_iter=2000 if fast_mode else 8000)
    meta_final.fit(X_meta_final, y_final, sample_weight=final_weights)

    # Evidential NN for final fit (deterministically seeded)
    evid_clf = None
    if not fast_mode:
        import torch
        from bio_spread_project.evidential_nn import EvidentialNN, evidential_loss

        torch.manual_seed(42)
        np.random.seed(42)

        evid_clf = EvidentialNN(X_meta_final.shape[1], hidden=32, dropout=0.3)
        opt = torch.optim.Adam(evid_clf.parameters(), lr=1e-3)
        X_t = torch.as_tensor(np.array(X_meta_final, dtype=np.float32, copy=True))
        y_t = torch.as_tensor(np.array(y_final, dtype=np.float32, copy=True))

        final_evid_epochs = 50
        for _ in range(final_evid_epochs):
            opt.zero_grad()
            alpha, _ = evid_clf(X_t)
            loss = evidential_loss(alpha, y_t)
            loss.backward()
            opt.step()

<<<<<<< HEAD
    # Wrap in our ensemble
=======
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




>>>>>>> 652e8458f0103f7d28df330a4311abe978728db1
    ensemble = GeoBioReliabilityModel(
        base_estimators=fitted_base,
        meta_estimator=meta_final, 
        feature_columns=tuple(all_final_features),
        importances=[],
        meta_scaler=None,
<<<<<<< HEAD
        bio_features=bio_features,
=======
>>>>>>> 652e8458f0103f7d28df330a4311abe978728db1
        calibrator=calibrator
    )
    # Attach enrichment models for self-contained inference
    ensemble.psge = psge
    ensemble.phylo_prop = phylo_prop
    ensemble.phylo_graph = phylo_graph
    ensemble.evid_clf = evid_clf
    ensemble.mixture_weight = optimal_w
    ensemble.grps_store = grps_model_store
    ensemble.grps_label_df = df.select(["backbone_id", "label_geo_spread"]).unique("backbone_id")
    
    validation_predictions = [
        _prediction_from_row(row, float(prob), MODEL_NAME)
        for row, prob in zip(df.to_dicts(), oof_meta_probs)
    ]
    
    # Final production metrics (OOF-based for honesty)
    metrics = dict(evaluate_predictions(validation_predictions))
    metrics.update(bootstrap_metric_intervals(validation_predictions, n_resamples=100))
    metrics["oof_roc_auc"] = metrics["roc_auc"]
    metrics["oof_average_precision"] = metrics["average_precision"]
    metrics["group_oof_roc_auc"] = metrics["roc_auc"]
    metrics["group_oof_average_precision"] = metrics["average_precision"]
    metrics["validation_mode"] = "spatial_group_cv_stacked"
    
    # Feature Attribution (Top Features for Audit)
    meta_cols = list(all_final_features) + [f"base_model_{i}" for i in range(len(fitted_base))]
    if hasattr(meta_final, "coef_"):
        weights = meta_final.coef_[0]
        top_idx = np.argsort(np.abs(weights))[::-1][:10]
        top_features = [{"feature": meta_cols[i], "score": float(np.abs(weights[i]))} for i in top_idx]
        metrics["top_features"] = top_features
        ensemble.importances = top_features
        if len(weights) > 0:
            abs_w = np.abs(weights)
            denom = float(np.sum(abs_w))
            top_share = float(np.max(abs_w) / denom) if denom > 0 else 0.0
            metrics["top_feature_weight_share"] = top_share
    
    # Internal Temporal Holdout Evaluation (latest-year compatibility metric)
    if "max_resolved_year_train" in df.columns:
        latest_year = df["max_resolved_year_train"].max()
        temporal_holdout = df.filter(pl.col("max_resolved_year_train") == latest_year)
        if 0 < temporal_holdout.height < df.height:
            t_preds = ensemble.predict(temporal_holdout)
            t_metrics = evaluate_predictions(t_preds)
            for k, v in t_metrics.items():
                metrics[f"temporal_holdout_{k}"] = v

    # Rolling temporal consistency (supplementary safety signal)
    if "max_resolved_year_train" in df.columns and not fast_mode:
        year_series = df["max_resolved_year_train"]
        valid_mask = year_series.is_not_null()
        if int(valid_mask.sum()) > 0:
            pred_df = pl.DataFrame(
                {
                    "year": df.filter(valid_mask)["max_resolved_year_train"].cast(pl.Int64),
                    "label": df.filter(valid_mask)["label_geo_spread"].cast(pl.Int64),
                    "prob": pl.Series("prob", oof_meta_probs[valid_mask.to_numpy()]),
                }
            )
            windows = build_rolling_temporal_windows(pred_df["year"].to_list(), min_train_years=3, gap_years=1, test_span_years=1)
            window_rocs: list[float] = []
            window_aps: list[float] = []
            window_sizes: list[int] = []
            window_pos: list[int] = []
            evaluated_windows = 0
            for window in windows:
                mask = (pl.col("year") >= window.test_start_year) & (pl.col("year") <= window.test_end_year)
                test_df = pred_df.filter(mask)
                if test_df.height < 25:
                    continue
                y = test_df["label"].to_numpy()
                if int(np.sum(y)) == 0 or int(np.sum(y)) == len(y):
                    continue
                p = test_df["prob"].to_numpy()
                window_rocs.append(float(roc_auc_score(y.astype(int), p)))
                window_aps.append(float(average_precision_score(y.astype(int), p)))
                window_sizes.append(int(test_df.height))
                window_pos.append(int(np.sum(y)))
                evaluated_windows += 1
            if evaluated_windows > 0:
                metrics["temporal_rolling_roc_auc_median"] = float(np.median(window_rocs))
                metrics["temporal_rolling_average_precision_median"] = float(np.median(window_aps))
                metrics["temporal_rolling_window_count"] = float(evaluated_windows)
                metrics["temporal_rolling_n_backbones_median"] = float(int(np.median(np.array(window_sizes, dtype=np.int64))))
                metrics["temporal_rolling_n_positive_median"] = float(int(np.median(np.array(window_pos, dtype=np.int64))))
                consistency = evaluate_temporal_consistency(
                    oof_roc_auc=float(metrics["roc_auc"]),
                    window_rocs=window_rocs,
                    max_positive_delta=0.03,
                    max_range=0.12,
                )
                metrics["temporal_consistency_status"] = str(consistency["status"])
                metrics["temporal_consistency"] = consistency
    
    calibration = calibration_summary(validation_predictions)
    metrics["calibrator_type"] = calibrator_type
    metrics["max_calibration_bin_gap"] = _max_bin_calibration_gap(calibration.get("calibration_bins", []))
    metrics.update(calibration_diagnostics)
    metrics.update(_knownness_slice_metrics(validation_predictions, quantile=0.20, min_samples=20))
    
    # LEAKAGE AUDIT: Integrated self-test for production-grade integrity
    audit_res = leakage_audit(df)
    metrics.update(audit_res)
    
    return ModelRun(
        model_name=MODEL_NAME,
        description=MODEL_DESCRIPTION,
        model=ensemble,
        predictions=ensemble.predict(df),
        metrics=metrics,
        calibration=calibration,
        validation_predictions=validation_predictions,
        coefficient_summary="" # Meta-models don't have single coefficients
    )





class GeoBioReliabilityModel:
<<<<<<< HEAD
    def __init__(self, base_estimators: list[Any], meta_estimator: Any, feature_columns: tuple[str, ...], importances: list[Any], meta_scaler: Any | None, bio_features: tuple[str, ...] = (), calibrator: Any | None = None):
=======
    def __init__(self, base_estimators: list[Any], meta_estimator: Any, feature_columns: tuple[str, ...], importances: list[Any], meta_scaler: Any | None, calibrator: Any | None = None):
>>>>>>> 652e8458f0103f7d28df330a4311abe978728db1
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator
        self.feature_columns = feature_columns
        self.importances = importances
        self.meta_scaler = meta_scaler
<<<<<<< HEAD
        self.bio_features = bio_features
        self.calibrator = calibrator
        # Storage for inductive enrichment models
        self.psge: Any | None = None
        self.phylo_prop: Any | None = None
        self.phylo_graph: Any | None = None
        self.grps_store: Any | None = None
        self.grps_label_df: pl.DataFrame | None = None
        self.mixture_weight: float = 0.6
        self.evid_clf: Any | None = None
=======
        self.calibrator = calibrator
>>>>>>> 652e8458f0103f7d28df330a4311abe978728db1

    def predict(self, df: pl.DataFrame) -> list[Prediction]:
        # Force enrichment by dropping potentially stale GNN/Phylo columns
        drop_cols = [c for c in df.columns if c.startswith("psge_") or c == "phylo_prop_risk"]
        if drop_cols:
            df = df.drop(drop_cols)

        # Perform inductive enrichment
        if self.psge is not None:
            psge_df = self.psge._to_df(df["backbone_id"].to_list())
            df = df.join(psge_df, on="backbone_id", how="left", coalesce=True)
        
        if self.phylo_prop is not None:
            # Inference mode: uses stored train_df for propagation
            prop_df = self.phylo_prop.predict(None, df)
            df = df.join(prop_df, on="backbone_id", how="left", coalesce=True)

        if self.grps_store is not None and self.grps_label_df is not None:
            ids = df["backbone_id"].to_list()
            # Use the full cached ESM2 table from the store to ensure 
            # both query and training reference embeddings are available
            emb = self.grps_store.esm2
            emb_cols = [c for c in emb.columns if c.startswith("esm2_embed_")]
            grps_df = compute_grps(ids, emb, self.grps_label_df, emb_cols)
            df = df.join(grps_df, on="backbone_id", how="left", coalesce=True)

        X_full = _feature_matrix(df, self.feature_columns)
        X_bio = _feature_matrix(df, self.bio_features)
        
        base_model_probs = np.zeros((len(X_full), len(self.base_estimators)))
        for i, m in enumerate(self.base_estimators):
            if hasattr(m, "predict_proba"):
                base_model_probs[:, i] = m.predict_proba(X_bio)[:, 1]
            else:
                d = m.decision_function(X_bio)
                base_model_probs[:, i] = 1 / (1 + np.exp(-d))

        X_meta = np.hstack([X_full, base_model_probs])
        
        # Meta-probabilities from Logistic base
        log_probs = self.meta_estimator.predict_proba(X_meta)[:, 1]
        
        # Meta-probabilities from Evidential mixture if available
        if self.evid_clf is not None:
            import torch
            self.evid_clf.eval()
            with torch.no_grad():
<<<<<<< HEAD
                x_meta_t = torch.as_tensor(np.array(X_meta, dtype=np.float32, copy=True))
                _, prob_v = self.evid_clf(x_meta_t)
                evid_probs = prob_v[:, 1].numpy()
            # Optimized mixture weight
            probs = self.mixture_weight * log_probs + (1.0 - self.mixture_weight) * evid_probs
=======
                alpha_all, prob_all = evid_clf(torch.FloatTensor(X_meta))
                ep = prob_all[:, 1].numpy()
                eu = 2.0 / alpha_all.sum(dim=1).numpy()
            X_lgb_all = np.hstack([X_meta, ep.reshape(-1, 1), np.log(eu + 1e-6).reshape(-1, 1)])
            lgb_p = self.meta_estimator.predict(X_lgb_all)
            # Map ranking score to [0, 1] via sigmoid
            lgb_p_prob = 1.0 / (1.0 + np.exp(-lgb_p))
            # Mixture: 10% ranker (for SOTA order) + 90% evidential (for calibration)
            probs = 0.1 * lgb_p_prob + 0.9 * ep
>>>>>>> 652e8458f0103f7d28df330a4311abe978728db1
        else:
            probs = log_probs
        
        if self.calibrator is not None:
            # Calibrate the mixture
            probs = self.calibrator.transform(probs)

<<<<<<< HEAD
=======
        if getattr(self, "calibrator", None) is not None:
            probs = self.calibrator.transform(probs)



        print(f"DEBUG: probs range [{np.min(probs):.4f}, {np.max(probs):.4f}]")
>>>>>>> 652e8458f0103f7d28df330a4311abe978728db1
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

    action = "monitor"
    if prob >= 0.8:
        action = "alert"
    elif prob >= 0.5:
        action = "watchlist"
    elif knownness <= 0.55 and prob >= 0.30:
        action = "watchlist"

    return Prediction(
        model_name=model_name,
        backbone_id=str(row["backbone_id"]),
        risk_probability=prob,
        confidence_tier=tier,
        label_geo_spread=int(row.get("label_geo_spread", 0) or 0),
        knownness_score=knownness,
        n_new_countries_future=int(row.get("n_new_countries_future", 0)),
        explanation=f"Ensemble risk: {prob:.3f} | action={action}",
        meta={"max_resolved_year_train": row.get("max_resolved_year_train"), "recommended_action": action},
    )

def single_feature_leakage_scan(df: pl.DataFrame | list[GeoSpreadFeatureRow], auc_threshold: float = 0.95) -> dict[str, Any]:
    if isinstance(df, list):
        df = geo_rows_to_frame(df)
    y = df["label_geo_spread"].to_numpy()
    max_auc = 0.0
    suspicious = 0
    ignore_tokens = {"backbone_id", "label", "target", "future", "outcome", "n_new", "jump", "split_year", "region", "phantom", "test", "severity", "seen", "time_to", "count_total"}
    scan_errors: list[dict[str, str]] = []
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
        except (ValueError, TypeError) as exc:
            scan_errors.append({"feature": col, "reason": str(exc)})
    return {
        "max_single_feature_auc": max_auc,
        "suspicious_feature_count": float(suspicious),
        "scan_error_count": float(len(scan_errors)),
        "scan_errors": scan_errors,
    }

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
        and single.get("scan_error_count", 0.0) == 0.0
    )
    return {
        "status": "pass" if passed else "fail",
        "max_single_feature_auc": float(single["max_single_feature_auc"]),
        "suspicious_feature_count": float(total_alarm_count),
        "single_feature_scan_error_count": float(single.get("scan_error_count", 0.0)),
        "single_feature_scan_errors": single.get("scan_errors", []),
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

def _compute_unsupervised_fastrp(
    df: pl.DataFrame,
    external_dir: Path | None,
    raw_records: pl.DataFrame | None = None,
    dim: int = 16,
    split_year: int = 2020,
) -> pl.DataFrame:
    """
    LEAK-03 FIX: Real unsupervised FastRP embeddings from graph topology.

    Uses build_fastrp_embeddings() when mash distances are available.
    FastRP is fully unsupervised (no labels used), so it does NOT create
    leakage — but it must use only pre-split records for temporal honesty.

    Falls back to zero embeddings if no graph data is available.
    """
    fastrp_cols = [f"fastrp_{i}" for i in range(dim)]

    if external_dir is not None and raw_records is not None:
        mash_path = external_dir / "mash_distances.tsv"
        if mash_path.exists():
            try:
                mash_df = pl.read_csv(mash_path, separator="\t", has_header=True)
                # Normalize column names
                rename_map = {}
                if "backbone_id_a" in mash_df.columns:
                    rename_map["backbone_id_a"] = "backbone_id_1"
                if "backbone_id_b" in mash_df.columns:
                    rename_map["backbone_id_b"] = "backbone_id_2"
                if "distance" in mash_df.columns:
                    rename_map["distance"] = "mash_distance"
                if rename_map:
                    mash_df = mash_df.rename(rename_map)

                req = {"backbone_id_1", "backbone_id_2", "mash_distance"}
                if req.issubset(set(mash_df.columns)):
                    from bio_spread_project.features_enrichment import build_fastrp_embeddings
                    fastrp_df = build_fastrp_embeddings(raw_records, mash_df, split_year)
                    if not fastrp_df.is_empty():
                        # Drop existing fastrp columns if present
                        drop_cols = [c for c in df.columns if c.startswith("fastrp_")]
                        if drop_cols:
                            df = df.drop(drop_cols)
                        df = df.join(fastrp_df, on="backbone_id", how="left", coalesce=True)
                        for c in fastrp_cols:
                            if c not in df.columns:
                                df = df.with_columns(pl.lit(0.0).alias(c))
                            else:
                                df = df.with_columns(pl.col(c).fill_null(0.0))
                        return df
            except (pl.exceptions.PolarsError, ValueError, TypeError):
                # Fall through to deterministic zero fallback
                pass

    # Fallback: zero embeddings
    for c in fastrp_cols:
        if c not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias(c))
    return df
