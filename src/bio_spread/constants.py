from __future__ import annotations

SNAPSHOT_FEATURE_COLS: list[str] = [
    "n_countries",
    "n_hosts",
    "years_since_first",
    "new_countries_recent",
    "new_countries_2y_ago",
    "n_records",
    "acceleration",
    "spread_velocity_norm",
    "niche_breadth",
]

STATIC_COLS: list[str] = [
    "log_size",
    "gc",
    "n_replicon_types",
    "has_relaxase",
    "n_relaxase_types",
    "mobility_score",
    "is_conjugative",
    "is_mobilizable",
    "topology",
    "has_orit",
    "n_orit_types",
    "host_range_rank",
]

CATEGORICAL_COLS: list[str] = [
    "replicon_types",
    "relaxase_types",
    "mpf_type",
    "plasmidfinder_dominant_type",
    "predicted_host_range_overall_name",
    "ecosystem_tags",
    "disease_tags",
]

TAXONOMY_COLS: list[str] = [
    "phylum_idx",
    "class_idx",
    "order_idx",
    "family_idx",
    "genus_idx",
]

TAXONOMY_RAW_COLS: list[str] = [
    "TAXONOMY_phylum",
    "TAXONOMY_class",
    "TAXONOMY_order",
    "TAXONOMY_family",
    "genus",
]

HEAVY_TAILED_FEATURES: set[str] = {
    "n_countries",
    "n_records",
    "new_countries_recent",
    "new_countries_2y_ago",
}

SNAPSHOT_NAN_COLS: list[str] = [
    f"{c}_nan_indicator" for c in SNAPSHOT_FEATURE_COLS
]

ALL_SNAPSHOT_COLS: list[str] = SNAPSHOT_FEATURE_COLS + SNAPSHOT_NAN_COLS

HAZARD_COLS: list[str] = ["hazard_1", "hazard_2", "hazard_3"]

COUNT_COL: str = "n_new_countries"
