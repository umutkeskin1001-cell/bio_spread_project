"""
Sovereign-X: Temporal snapshot builder with hazard targets.

Key change from V7: targets are now hazard probabilities for years 1, 2, 3
and backbone-disjoint splits replace temporal-only splits.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# Columns exported for use by dataset module
# Time-varying snapshot features — purely epidemiological, no static overlap
SNAPSHOT_FEATURE_COLS = [
    "n_countries",
    "n_hosts",
    "years_since_first",
    "new_countries_recent",
    "new_countries_2y_ago",
    "n_records",
    "acceleration",
    "expansion_ratio",
    "spread_velocity",
    "niche_breadth",
]
# Static backbone-level features — time-invariant biological properties
STATIC_COLS = [
    "log_size",
    "gc",
    "n_replicon_types",
    "n_relaxase_types",
    "mobility_score",
    "is_conjugative",
    "is_mobilizable",
    "topology",
    "n_orit_types",
    "host_range_rank",
]
TAXONOMY_COLS = ["phylum_idx", "class_idx", "order_idx", "family_idx", "genus_idx"]


def _nonnull(values: list[str | None]) -> list[str]:
    return [v for v in values if v is not None and str(v).strip()]


def _unique_nonnull(values: list[str | None]) -> set[str]:
    return set(_nonnull(values))


def build_taxonomy_vocab(df: pl.DataFrame) -> dict[str, dict[str, int]]:
    """Build integer vocabularies from taxonomy columns in the raw DataFrame.

    Returns nested dict:
        {"TAXONOMY_phylum": {"Pseudomonadota": 1, ...}, "TAXONOMY_class": {...}, ...}
    Index 0 is reserved for "unknown/UNKNOWN".
    Returns empty dict if taxonomy columns not found.
    """
    tax_columns = ["TAXONOMY_phylum", "TAXONOMY_class", "TAXONOMY_order", "TAXONOMY_family", "genus"]
    missing = [c for c in tax_columns if c not in df.columns]
    if missing:
        logger.warning("Taxonomy columns missing: %s. Skipping taxonomy encoding.", missing)
        return {}

    vocab = {}
    for col in tax_columns:
        values = df[col].drop_nulls().unique().to_list()
        values = [str(v).strip() for v in values if v is not None and str(v).strip()]
        # Sort for deterministic ordering
        values = sorted(set(values))
        mapping = {"UNKNOWN": 0}
        mapping.update({v: i + 1 for i, v in enumerate(values)})
        vocab[col] = mapping
    logger.info(
        "Built taxonomy vocab: phyla=%d, classes=%d, orders=%d, families=%d, genera=%d",
        len(vocab.get("TAXONOMY_phylum", {})),
        len(vocab.get("TAXONOMY_class", {})),
        len(vocab.get("TAXONOMY_order", {})),
        len(vocab.get("TAXONOMY_family", {})),
        len(vocab.get("genus", {})),
    )
    return vocab


def get_dominant_taxonomy(
    df: pl.DataFrame, backbone_id: str, vocab: dict[str, dict[str, int]], tax_columns: list[str]
) -> dict[str, int]:
    """Get the most common taxonomy value for a backbone at each level.
    Returns indices into the vocab."""
    rows = df.filter(pl.col("backbone_id") == backbone_id)
    result = {}
    for col in tax_columns:
        mapping = vocab.get(col, {})
        if rows.is_empty():
            result[col] = 0
        else:
            vals = rows[col].drop_nulls().to_list()
            vals = [str(v).strip() for v in vals if v is not None and str(v).strip()]
            if not vals:
                result[col] = 0
            else:
                # Mode (most common value)
                mode_val = Counter(vals).most_common(1)[0][0]
                result[col] = mapping.get(mode_val, 0)
    return result


def save_taxonomy_vocab(vocab: dict, path: Path):
    with open(path, "w") as f:
        json.dump(vocab, f, indent=2)


def load_taxonomy_vocab(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


class FeatureBuilder:
    """Computes static + temporal features from raw plasmid records.

    Stateless: all data passed in, pure computation out.
    """

    def __init__(self, horizon: int = 3, require_country_history: bool = True):
        self.horizon = horizon
        self.require_country_history = require_country_history

    def static_features(self, meta: pl.DataFrame) -> pl.DataFrame:
        """Per-backbone time-invariant features."""
        # Compute mobility score from meta before select (to avoid column loss)
        if "predicted_mobility" in meta.columns:
            mobility_expr = (
                pl.col("predicted_mobility")
                .fill_null("non-mobilizable")
                .map_elements(
                    lambda x: {"mobilizable": 1, "conjugative": 2}.get(str(x), 0),
                    return_dtype=pl.Float64,
                )
                .alias("mobility_score")
            )
        else:
            mobility_expr = pl.lit(0.0, dtype=pl.Float64).alias("mobility_score")

        # Enrich static features with backbone-level biologically informative attributes
        # is_conjugative, is_mobilizable, topology: 100% coverage from meta
        # n_orit_types: 100% coverage (0-3 range)
        # host_range_rank: 86.9% coverage (1-7, filled to 0 for unknown)
        if "is_conjugative" in meta.columns:
            conj_expr = pl.col("is_conjugative").cast(pl.Int64).fill_null(0).cast(pl.Float64).alias("is_conjugative")
        else:
            conj_expr = pl.lit(0.0, dtype=pl.Float64).alias("is_conjugative")

        if "is_mobilizable" in meta.columns:
            mob_expr = pl.col("is_mobilizable").cast(pl.Int64).fill_null(0).cast(pl.Float64).alias("is_mobilizable")
        else:
            mob_expr = pl.lit(0.0, dtype=pl.Float64).alias("is_mobilizable")

        if "topology" in meta.columns:
            topo_expr = (
                pl.col("topology")
                .fill_null("circular")
                .map_elements(lambda x: 1.0 if str(x).strip().lower() == "linear" else 0.0, return_dtype=pl.Float64)
                .alias("topology")
            )
        else:
            topo_expr = pl.lit(0.0, dtype=pl.Float64).alias("topology")

        if "n_orit_types" in meta.columns:
            orit_expr = pl.col("n_orit_types").cast(pl.Int64).fill_null(0).cast(pl.Float64).alias("n_orit_types")
        else:
            orit_expr = pl.lit(0.0, dtype=pl.Float64).alias("n_orit_types")

        if "predicted_host_range_overall_rank" in meta.columns:
            _rank_map = {"species": 0, "genus": 1, "family": 2, "order": 3, "class": 4, "phylum": 5, "multi-phylla": 6}
            host_r_expr = (
                pl.col("predicted_host_range_overall_rank")
                .fill_null("")
                .map_elements(lambda x: float(_rank_map.get(str(x).strip().lower(), 0.0)), return_dtype=pl.Float64)
                .alias("host_range_rank")
            )
        else:
            host_r_expr = pl.lit(0.0, dtype=pl.Float64).alias("host_range_rank")

        df = meta.unique(subset=["backbone_id"]).select(
            [
                "backbone_id",
                pl.col("size").fill_null(0).cast(pl.Float64).log1p().alias("log_size"),
                pl.col("gc").fill_null(0.5).cast(pl.Float64),
                pl.col("n_replicon_types").fill_null(0).cast(pl.Float64),
                pl.col("n_relaxase_types").fill_null(0).cast(pl.Float64),
                mobility_expr,
                conj_expr,
                mob_expr,
                topo_expr,
                orit_expr,
                host_r_expr,
            ]
        )
        return df

    def backcast_features(self, history: pl.DataFrame, cutoff_year: int) -> Dict[str, float]:
        """Compute epidemiologic features using ONLY data <= cutoff_year."""
        if history.is_empty():
            return self._zero_features()

        first_year = history["year"].min()
        countries = _unique_nonnull(history["country"].to_list() if "country" in history.columns else [])
        hosts = _unique_nonnull(history["host_genus"].to_list() if "host_genus" in history.columns else [])

        n_countries = len(countries)
        n_hosts = len(hosts)
        years_since = float(cutoff_year - first_year)

        # Velocity: new countries in last 2 years
        last_2y = history.filter(pl.col("year") >= (cutoff_year - 2))
        older = history.filter(pl.col("year") < (cutoff_year - 2))
        new_recent = len(
            _unique_nonnull(last_2y["country"].to_list() if "country" in last_2y.columns else [])
            - _unique_nonnull(older["country"].to_list() if "country" in older.columns else [])
        )

        # Velocity 2 years ago
        last_4y_to_2y = history.filter((pl.col("year") >= (cutoff_year - 4)) & (pl.col("year") < (cutoff_year - 2)))
        older_than_4y = history.filter(pl.col("year") < (cutoff_year - 4))
        new_2y_ago = len(
            _unique_nonnull(last_4y_to_2y["country"].to_list() if "country" in last_4y_to_2y.columns else [])
            - _unique_nonnull(older_than_4y["country"].to_list() if "country" in older_than_4y.columns else [])
        )

        countries_2y_ago = len(
            _unique_nonnull(older_than_4y["country"].to_list() if "country" in older_than_4y.columns else [])
            | _unique_nonnull(last_4y_to_2y["country"].to_list() if "country" in last_4y_to_2y.columns else [])
        )

        return {
            "n_countries": float(n_countries),
            "n_hosts": float(n_hosts),
            "years_since_first": years_since,
            "new_countries_recent": float(new_recent),
            "new_countries_2y_ago": float(new_2y_ago),
            "n_records": float(len(history)),
            "acceleration": float(new_recent - new_2y_ago),
            "expansion_ratio": float(n_countries) / max(float(countries_2y_ago), 1.0),
            "spread_velocity": float(n_countries) / max(years_since, 1.0),
            "niche_breadth": float(n_hosts) / max(float(n_countries), 1.0),
        }

    def hazard_targets(
        self, cutoff_year: int, max_year: int,
        country_progression: dict[int, set[str]] | None = None,
    ) -> Dict[str, float]:
        """Hazard: P(spread within years 1, 2, 3 after cutoff).

        Uses pre-computed country progression dict for O(1) lookups
        instead of re-filtering the full DataFrame per (backbone, year).

        Args:
            cutoff_year: the "present" year for this snapshot.
            max_year: latest year in the full dataset (for censoring detection).
            country_progression: dict year → set of countries seen up to that year.
                If None, all targets are -1 (censored).

        Returns:
            Dict with hazard_1, hazard_2, hazard_3, n_new_countries, observed.
        """
        targets: Dict[str, float] = {}

        if country_progression is None or cutoff_year not in country_progression:
            for h in range(1, self.horizon + 1):
                targets[f"hazard_{h}"] = -1.0
            targets["n_new_countries"] = -1.0
            targets["observed"] = 0.0
            return targets

        past = country_progression[cutoff_year]

        for horizon in range(1, self.horizon + 1):
            window_end = int(cutoff_year) + horizon
            if window_end > max_year:
                targets[f"hazard_{horizon}"] = -1.0  # right-censored
                continue
            future_until = country_progression.get(window_end, set())
            n_new = len(future_until - past)
            targets[f"hazard_{horizon}"] = float(n_new > 0)

        # Count of new countries in full horizon
        window_end_3 = int(cutoff_year) + self.horizon
        if window_end_3 > max_year:
            targets["n_new_countries"] = -1.0
        else:
            future_3 = country_progression.get(window_end_3, set())
            targets["n_new_countries"] = float(len(future_3 - past))

        is_observed = all(v >= 0 for v in targets.values())
        targets["observed"] = float(is_observed)
        return targets

    def _zero_features(self) -> Dict[str, float]:
        return {
            "n_countries": 0.0,
            "n_hosts": 0.0,
            "years_since_first": 0.0,
            "new_countries_recent": 0.0,
            "new_countries_2y_ago": 0.0,
            "n_records": 0.0,
            "acceleration": 0.0,
            "expansion_ratio": 0.0,
            "spread_velocity": 0.0,
            "niche_breadth": 0.0,
        }


def build_sequences(
    raw: pl.DataFrame,
    meta: pl.DataFrame,
    backbone_ids: set[str],
    horizon: int = 3,
    min_snapshots: int = 2,
    require_country_history: bool = True,
    taxonomy_vocab: dict | None = None,
) -> pl.DataFrame:
    """Build backbone-level snapshot sequences. Returns one row per (backbone_id, year).

    Each backbone's snapshots are sorted by year. Only backbones with
    >= min_snapshots observations are kept.

    If taxonomy_vocab is provided, adds taxonomy index columns.
    """
    builder = FeatureBuilder(horizon, require_country_history)
    max_year = int(raw["year"].max()) if not raw.is_empty() else 0

    # Static features
    static = builder.static_features(meta)
    static_dict = {row["backbone_id"]: row for row in static.to_dicts()}

    # Pre-compute taxonomy indices if vocab provided
    tax_columns_raw = ["TAXONOMY_phylum", "TAXONOMY_class", "TAXONOMY_order", "TAXONOMY_family", "genus"]
    use_taxonomy = taxonomy_vocab is not None and bool(taxonomy_vocab)
    taxonomy_cache: dict[str, dict[str, int]] = {}
    if use_taxonomy:
        for bid in backbone_ids:
            taxonomy_cache[bid] = get_dominant_taxonomy(raw, bid, taxonomy_vocab, tax_columns_raw)

    # Pre-compute country progression per backbone for O(1) hazard lookups
    # country_cache[backbone_id] = { year: set_of_countries_seen_up_to_year }
    country_cache: dict[str, dict[int, set[str]]] = {}
    for bid in backbone_ids:
        bid_raw = raw.filter(pl.col("backbone_id") == bid).sort("year")
        bid_years = bid_raw["year"].unique().to_list()
        if len(bid_years) < min_snapshots:
            continue
        progression: dict[int, set[str]] = {}
        seen: set[str] = set()
        for year in bid_years:
            year_countries = _unique_nonnull(
                bid_raw.filter(pl.col("year") == year)["country"].to_list()
                if "country" in bid_raw.columns
                else []
            )
            seen = seen | year_countries
            progression[year] = seen
        country_cache[bid] = progression

    rows = []
    for bid in backbone_ids:
        bid_raw = raw.filter(pl.col("backbone_id") == bid).sort("year")
        bid_years = bid_raw["year"].unique().to_list()
        if len(bid_years) < min_snapshots:
            continue

        progression = country_cache.get(bid, {})
        s = static_dict.get(bid, {})
        tax = taxonomy_cache.get(bid, {})

        for year in bid_years:
            history = bid_raw.filter(pl.col("year") <= year)
            bf = builder.backcast_features(history, year)
            ht = builder.hazard_targets(year, max_year, country_progression=progression)

            row = {
                "backbone_id": bid,
                "year": year,
                "log_size": s.get("log_size", 0.0),
                "gc": s.get("gc", 0.5),
                "n_replicon_types": s.get("n_replicon_types", 0.0),
                "n_relaxase_types": s.get("n_relaxase_types", 0.0),
                "mobility_score": s.get("mobility_score", 0.0),
                "is_conjugative": float(s.get("is_conjugative", 0.0)),
                "is_mobilizable": float(s.get("is_mobilizable", 0.0)),
                "topology": float(s.get("topology", 0.0)),
                "n_orit_types": float(s.get("n_orit_types", 0.0)),
                "host_range_rank": float(s.get("host_range_rank", 0.0)),
                **bf,
                "hazard_1": ht["hazard_1"],
                "hazard_2": ht["hazard_2"],
                "hazard_3": ht["hazard_3"],
                "n_new_countries": ht["n_new_countries"],
                "observed": ht["observed"],
            }
            if use_taxonomy:
                row["phylum_idx"] = tax.get("TAXONOMY_phylum", 0)
                row["class_idx"] = tax.get("TAXONOMY_class", 0)
                row["order_idx"] = tax.get("TAXONOMY_order", 0)
                row["family_idx"] = tax.get("TAXONOMY_family", 0)
                row["genus_idx"] = tax.get("genus", 0)

            rows.append(row)

    result = pl.DataFrame(rows)
    # Enforce dtypes
    casts = {}
    for col in result.columns:
        if col in ("backbone_id",):
            continue
        if result[col].dtype == pl.Utf8:
            casts[col] = pl.Float64
        elif result[col].dtype in (pl.Int32, pl.Int64):
            casts[col] = pl.Int64
    if casts:
        result = result.with_columns([pl.col(c).cast(dt) for c, dt in casts.items()])

    logger.info("Built %d sequences for %d backbones", len(result), result["backbone_id"].n_unique())
    return result


def disjoint_backbone_split(
    raw: pl.DataFrame, split_year: int, val_frac: float = 0.15, test_frac: float = 0.15
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Split backbones by first observation year. Clean leakage prevention."""
    first_year = raw.group_by("backbone_id").agg(pl.col("year").min().alias("first_year"))

    # Backbones first seen before split_year → train candidates
    train_candidates = set(first_year.filter(pl.col("first_year") < split_year)["backbone_id"].to_list())
    # Backbones first seen at or after split_year → test (cold-start) candidates
    test_candidates = set(first_year.filter(pl.col("first_year") >= split_year)["backbone_id"].to_list())

    # For val: sample from train candidates that have sufficient temporal coverage
    # This gives us temporal continuity evaluation
    rng = np.random.default_rng(42)
    train_list = sorted(train_candidates)
    rng.shuffle(train_list)
    n_val = max(1, int(len(train_list) * val_frac))
    val_ids = set(train_list[:n_val])
    train_ids = set(train_list[n_val:])

    # For test: sample from test candidates
    test_list = sorted(test_candidates)
    rng.shuffle(test_list)
    n_test = max(1, int(len(test_list) * (test_frac / (val_frac + test_frac))))
    test_ids = set(test_list[:n_test])
    # Add remaining test candidates back to training
    extra_train = set(test_list[n_test:])

    train_ids |= extra_train

    logger.info(
        "Disjoint split: %d train, %d val, %d test backbones",
        len(train_ids),
        len(val_ids),
        len(test_ids),
    )
    return train_ids, val_ids, test_ids
