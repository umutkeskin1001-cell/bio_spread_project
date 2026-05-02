import polars as pl

INTERACTION_PAIRS = [
    ("T_eff_norm", "H_obs_specialization_norm"),
    ("T_eff_norm", "A_eff_norm"),
    ("H_obs_specialization_norm", "A_eff_norm"),
    ("coherence_score", "country_slope_train"),
    ("H_external_host_range_norm", "replicon_architecture_norm"),
    ("orit_support", "amr_burden_saturation_norm"),
    ("backbone_purity_norm", "assignment_confidence_norm"),
    ("geo_country_entropy_train", "host_breadth_slope_train"),
    ("country_slope_train", "mobility_shift_slope_train"),
    ("mean_antibiotic_pressure", "frac_pathogenic_hosts"),
]

def build_synergy_features(
    features: pl.DataFrame,
    pairs: list[tuple[str, str]],
) -> pl.DataFrame:
    """
    For each (col_a, col_b) pair, create a new column 'synergy_{col_a}__{col_b}'
    equal to column_a * column_b. Missing columns are silently skipped.
    """
    for a, b in pairs:
        if a in features.columns and b in features.columns:
            name = f"synergy_{a}__{b}"
            features = features.with_columns(
                (pl.col(a) * pl.col(b)).alias(name)
            )
    return features
