from __future__ import annotations

import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import polars as pl

from bio_spread.constants import CATEGORICAL_COLS, TAXONOMY_RAW_COLS

logger = logging.getLogger(__name__)

MULTI_VALUE_COLS = {"replicon_types", "relaxase_types", "ecosystem_tags", "disease_tags"}

CAT_VOCAB_LIMITS = {
    "replicon_types": 100,
    "relaxase_types": 30,
    "mpf_type": 10,
    "plasmidfinder_dominant_type": 100,
    "predicted_host_range_overall_name": 100,
    "ecosystem_tags": 50,
    "disease_tags": 50,
}


def _nonnull(values: list[str | None]) -> list[str]:
    return [v for v in values if v is not None and str(v).strip()]


def _unique_nonnull(values: list[str | None]) -> set[str]:
    return set(_nonnull(values))


def build_taxonomy_vocab(df: pl.DataFrame) -> dict[str, dict[str, int]]:
    tax_columns = TAXONOMY_RAW_COLS
    missing = [c for c in tax_columns if c not in df.columns]
    if missing:
        logger.warning("Taxonomy columns missing: %s. Skipping taxonomy encoding.", missing)
        return {}

    vocab = {}
    for col in tax_columns:
        values = df[col].drop_nulls().unique().to_list()
        values = [str(v).strip() for v in values if v is not None and str(v).strip()]
        values = sorted(set(values))
        if "UNKNOWN" in values:
            raise ValueError(
                f"Taxonomy column '{col}' contains literal 'UNKNOWN' which conflicts with sentinel"
            )
        mapping = {}
        mapping["UNKNOWN"] = 1
        mapping.update({v: i + 2 for i, v in enumerate(values)})
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
    rows = df.filter(pl.col("backbone_id") == backbone_id)
    result = {}
    for col in tax_columns:
        mapping = vocab.get(col, {})
        if rows.is_empty():
            result[col] = 1
        else:
            vals = rows[col].drop_nulls().to_list()
            vals = [str(v).strip() for v in vals if v is not None and str(v).strip()]
            if not vals:
                result[col] = 1
            else:
                mode_val = Counter(vals).most_common(1)[0][0]
                result[col] = mapping.get(mode_val, 1)
    return result


def save_taxonomy_vocab(vocab: dict, path: Path):
    with open(path, "w") as f:
        json.dump(vocab, f, indent=2)


def load_taxonomy_vocab(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_categorical_vocabs(
    df: pl.DataFrame, cols: list[str] | None = None, max_vocab: int = 100
) -> dict[str, dict[str, int]]:
    if cols is None:
        cols = CATEGORICAL_COLS
    vocabs = {}
    for col in cols:
        if col not in df.columns:
            continue
        limit = CAT_VOCAB_LIMITS.get(col, max_vocab)
        values = df[col].drop_nulls().to_list()
        all_tokens = []
        for v in values:
            s = str(v).strip()
            if not s:
                continue
            if col in MULTI_VALUE_COLS:
                tokens = [t.strip() for t in s.split(",") if t.strip()]
                all_tokens.extend(tokens)
            else:
                all_tokens.append(s)
        counts = Counter(all_tokens)
        top = counts.most_common(limit)
        mapping = {"UNKNOWN": 0}
        for i, (token, _) in enumerate(top):
            mapping[token] = i + 1
        vocabs[col] = mapping
        logger.info("Categorical vocab '%s': %d tokens (top %d)", col, len(mapping), limit)
    return vocabs


def save_categorical_vocabs(vocabs: dict, path: Path):
    with open(path, "w") as f:
        json.dump(vocabs, f, indent=2)


def load_categorical_vocabs(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def encode_categorical_value(
    value: str | None, col: str, vocab: dict[str, int]
) -> List[int]:
    if value is None or not str(value).strip():
        return [0]
    s = str(value).strip()
    if col in MULTI_VALUE_COLS:
        tokens = [t.strip() for t in s.split(",") if t.strip()]
        indices = [vocab.get(t, 0) for t in tokens]
        return indices if indices else [0]
    else:
        return [vocab.get(s, 0)]


class FeatureBuilder:
    def __init__(self, horizon: int = 3, require_country_history: bool = True):
        self.horizon = horizon
        self.require_country_history = require_country_history

    def static_features(self, meta: pl.DataFrame) -> pl.DataFrame:
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

        if "has_orit" in meta.columns:
            has_orit_expr = pl.col("has_orit").cast(pl.Int64).fill_null(0).cast(pl.Float64).alias("has_orit")
        else:
            has_orit_expr = pl.lit(0.0, dtype=pl.Float64).alias("has_orit")

        if "has_relaxase" in meta.columns:
            has_relaxase_expr = pl.col("has_relaxase").cast(pl.Int64).fill_null(0).cast(pl.Float64).alias("has_relaxase")
        else:
            has_relaxase_expr = pl.lit(0.0, dtype=pl.Float64).alias("has_relaxase")

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
                has_orit_expr,
                has_relaxase_expr,
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
        if history.is_empty():
            return self._zero_features()

        first_year = history["year"].min()
        countries = _unique_nonnull(history["country"].to_list() if "country" in history.columns else [])
        n_countries = len(countries)
        years_since = float(cutoff_year - first_year)

        last_2y = history.filter(pl.col("year") >= (cutoff_year - 2))
        older = history.filter(pl.col("year") < (cutoff_year - 2))
        new_recent = len(
            _unique_nonnull(last_2y["country"].to_list() if "country" in last_2y.columns else [])
            - _unique_nonnull(older["country"].to_list() if "country" in older.columns else [])
        )

        last_4y_to_2y = history.filter((pl.col("year") >= (cutoff_year - 4)) & (pl.col("year") < (cutoff_year - 2)))
        older_than_4y = history.filter(pl.col("year") < (cutoff_year - 4))
        new_2y_ago = len(
            _unique_nonnull(last_4y_to_2y["country"].to_list() if "country" in last_4y_to_2y.columns else [])
            - _unique_nonnull(older_than_4y["country"].to_list() if "country" in older_than_4y.columns else [])
        )

        spread_velocity = float(new_recent) / max(years_since, 1.0)
        spread_norm = math.tanh(spread_velocity)

        n_hosts_val = 0.0
        if "host_genus" in history.columns:
            n_hosts_val = float(len(_unique_nonnull(history["host_genus"].to_list())))
        elif "n_hosts" in history.columns:
            n_hosts_val = float(history["n_hosts"].to_list()[-1])

        niche_breadth_val = 0.0
        if "niche_breadth" in history.columns:
            niche_breadth_val = float(history["niche_breadth"].to_list()[-1])

        return {
            "n_countries": float(n_countries),
            "n_hosts": n_hosts_val,
            "years_since_first": years_since,
            "new_countries_recent": float(new_recent),
            "new_countries_2y_ago": float(new_2y_ago),
            "n_records": float(len(history)),
            "acceleration": float(new_recent - new_2y_ago),
            "spread_velocity_norm": spread_norm,
            "niche_breadth": niche_breadth_val,
        }

    def hazard_targets(
        self, cutoff_year: int, max_year: int,
        country_progression: dict[int, set[str]] | None = None,
    ) -> Dict[str, float]:
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
                targets[f"hazard_{horizon}"] = -1.0
                continue
            future_until = country_progression.get(window_end, set())
            n_new = len(future_until - past)
            targets[f"hazard_{horizon}"] = float(n_new > 0)

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
            "spread_velocity_norm": 0.0,
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
    categorical_vocabs: dict | None = None,
) -> pl.DataFrame:
    builder = FeatureBuilder(horizon, require_country_history)

    static = builder.static_features(meta)
    static_dict = {row["backbone_id"]: row for row in static.to_dicts()}

    use_taxonomy = taxonomy_vocab is not None and bool(taxonomy_vocab)
    use_categorical = categorical_vocabs is not None and bool(categorical_vocabs)

    # Pre-compute backbone groups ONCE — replaces O(N*M) per-backbone filter
    raw_groups: dict[str, pl.DataFrame] = {
        bid: group.sort("year")
        for bid, group in raw.group_by("backbone_id", maintain_order=True)
        if bid in backbone_ids
    }

    # Taxonomy cache from pre-grouped data (no per-backbone filter)
    taxonomy_cache: dict[str, dict[str, int]] = {}
    if use_taxonomy:
        for bid, group in raw_groups.items():
            tax_result = {}
            for col in TAXONOMY_RAW_COLS:
                mapping = taxonomy_vocab.get(col, {})
                vals = group[col].drop_nulls().to_list()
                vals = [str(v).strip() for v in vals if v is not None and str(v).strip()]
                if vals:
                    mode_val = Counter(vals).most_common(1)[0][0]
                    tax_result[col] = mapping.get(mode_val, 1)
                else:
                    tax_result[col] = 1
            taxonomy_cache[bid] = tax_result

    # Categorical cache from pre-grouped meta
    cat_cache: dict[str, dict[str, list[int]]] = {}
    if use_categorical:
        meta_groups: dict[str, pl.DataFrame] = {
            bid: group
            for bid, group in meta.group_by("backbone_id", maintain_order=True)
            if bid in backbone_ids
        }
        for bid in backbone_ids:
            meta_group = meta_groups.get(bid)
            if meta_group is None or meta_group.is_empty():
                cat_cache[bid] = {}
                continue
            cat_cache[bid] = {}
            for col, vocab in categorical_vocabs.items():
                if col in meta_group.columns:
                    val = meta_group[col].to_list()[0]
                    cat_cache[bid][col] = encode_categorical_value(val, col, vocab)
                else:
                    cat_cache[bid][col] = [0]

    rows = []
    country_col = "country" if "country" in raw.columns else None

    for bid in backbone_ids:
        bid_raw = raw_groups.get(bid)
        if bid_raw is None:
            continue

        bid_years = bid_raw["year"].unique().to_list()
        if len(bid_years) < min_snapshots:
            continue

        # Build progression using index-based year slicing
        years_arr = bid_raw["year"].to_numpy()
        unique_years, year_counts = np.unique(years_arr, return_counts=True)
        year_ends = np.cumsum(year_counts)
        year_starts = np.concatenate([[0], year_ends[:-1]])

        progression: dict[int, set[str]] = {}
        seen: set[str] = set()
        if country_col:
            for i, year in enumerate(unique_years):
                year_countries = _unique_nonnull(bid_raw[country_col].to_list()[year_starts[i]:year_ends[i]])
                seen = seen | year_countries
                progression[year] = seen
        else:
            for year in unique_years:
                progression[year] = seen

        if progression:
            min_y, max_y = int(min(progression)), int(max(progression))
            last_seen = progression[min_y]
            for y in range(min_y, max_y + 1):
                if y in progression:
                    last_seen = progression[y]
                else:
                    progression[y] = last_seen

        bid_max_year = int(bid_raw["year"].max())
        s = static_dict.get(bid, {})
        tax = taxonomy_cache.get(bid, {})
        cat = cat_cache.get(bid, {})

        year_to_last_idx = {yr: i for i, yr in enumerate(bid_years)}
        year_count = len(bid_years)

        for yi, year in enumerate(bid_years):
            last_idx = year_to_last_idx[year]
            history = bid_raw[:last_idx + 1]
            bf = builder.backcast_features(history, year)
            ht = builder.hazard_targets(year, bid_max_year, country_progression=progression)

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
                "has_orit": float(s.get("has_orit", 0.0)),
                "has_relaxase": float(s.get("has_relaxase", 0.0)),
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
                row["phylum_idx"] = tax.get("TAXONOMY_phylum", 1)
                row["class_idx"] = tax.get("TAXONOMY_class", 1)
                row["order_idx"] = tax.get("TAXONOMY_order", 1)
                row["family_idx"] = tax.get("TAXONOMY_family", 1)
                row["genus_idx"] = tax.get("genus", 1)

            if use_categorical:
                for col in CATEGORICAL_COLS:
                    indices = cat.get(col, [0])
                    row[f"cat_{col}"] = json.dumps(indices)

            rows.append(row)

    result = pl.DataFrame(rows)
    casts = {}
    for col in result.columns:
        if col in ("backbone_id",):
            continue
        if col.startswith("cat_"):
            continue
        if result[col].dtype == pl.Utf8:
            casts[col] = pl.Float64
        elif result[col].dtype in (pl.Int32, pl.Int64):
            casts[col] = pl.Int64
    if casts:
        result = result.with_columns([pl.col(c).cast(dt) for c, dt in casts.items()])

    logger.info("Built %d sequences for %d backbones", len(result), result["backbone_id"].n_unique())
    return result


def random_backbone_split(
    raw: pl.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15, seed: int = 42
) -> Tuple[Set[str], Set[str], Set[str]]:
    all_bids = raw["backbone_id"].unique().to_list()
    if len(all_bids) < 4:
        raise ValueError(f"Need >= 4 backbones, got {len(all_bids)}")
    rng = np.random.default_rng(seed)
    bid_list = sorted(all_bids)
    rng.shuffle(bid_list)
    n = len(bid_list)
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))
    train_ids = set(bid_list[:n_train])
    val_ids = set(bid_list[n_train:n_train + n_val])
    test_ids = set(bid_list[n_train + n_val:])
    logger.info(
        "Random split: %d train, %d val, %d test backbones (test = truly unseen)",
        len(train_ids), len(val_ids), len(test_ids),
    )
    return train_ids, val_ids, test_ids


# Year-disjoint split for backward compatibility (backbones first seen >= split_year → test)
def disjoint_backbone_split(
    raw: pl.DataFrame, split_year: int = 2020, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42
) -> Tuple[Set[str], Set[str], Set[str]]:
    all_bids = raw["backbone_id"].unique().to_list()
    if len(all_bids) < 4:
        raise ValueError(f"Need >= 4 backbones, got {len(all_bids)}")

    first_years = raw.group_by("backbone_id").agg(pl.col("year").min())
    train_candidates = []
    test_bids_list = []
    for row in first_years.to_dicts():
        if row["year"] < split_year:
            train_candidates.append(row["backbone_id"])
        else:
            test_bids_list.append(row["backbone_id"])

    if not train_candidates:
        raise ValueError(f"No backbones first seen before {split_year}")
    test_bids = set(test_bids_list)

    rng = np.random.default_rng(seed)
    rng.shuffle(train_candidates)
    n_val = max(1, int(len(train_candidates) * val_frac))
    val_ids = set(train_candidates[:n_val])
    train_ids = set(train_candidates[n_val:])

    logger.info(
        "Disjoint split: %d train, %d val, %d test (split_year=%d, pre-%d=%d, post=%d)",
        len(train_ids), len(val_ids), len(test_bids),
        split_year, split_year, len(train_candidates), len(test_bids),
    )
    return train_ids, val_ids, test_bids
