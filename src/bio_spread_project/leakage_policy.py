from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl


FORBIDDEN_TOKENS = ("future", "target", "outcome", "event_within_", "time_to_", "jump")
ALLOWLIST_EXACT = {"label_geo_spread", "n_new_countries_future"}
NON_MODEL_COLUMNS = {
    "backbone_id",
    "region",
    "knownness_score",
    "max_resolved_year_train",
}


def load_feature_lineage(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("features", {})
    if not isinstance(entries, dict):
        raise ValueError("feature_lineage config must contain a 'features' object")
    out: dict[str, dict[str, Any]] = {}
    for k, v in entries.items():
        if isinstance(v, dict):
            out[str(k)] = dict(v)
    return out


def detect_forbidden_columns(columns: list[str]) -> list[str]:
    blocked: list[str] = []
    for col in columns:
        lowered = col.lower()
        if col not in ALLOWLIST_EXACT and any(token in lowered for token in FORBIDDEN_TOKENS):
            blocked.append(col)
    return sorted(set(blocked))


def validate_feature_surface(
    features: pl.DataFrame,
    *,
    lineage_path: Path | None = None,
    strict_lineage: bool = False,
) -> dict[str, Any]:
    columns = [str(c) for c in features.columns]
    blocked: list[str] = detect_forbidden_columns(columns)
    unknown_lineage: list[str] = []
    lineage = load_feature_lineage(lineage_path) if lineage_path is not None else {}

    for col in columns:
        if col in NON_MODEL_COLUMNS or col.endswith("_right"):
            continue
        if lineage and col not in lineage and col not in ALLOWLIST_EXACT:
            unknown_lineage.append(col)

    if blocked:
        raise ValueError(f"Leakage policy violation: forbidden feature tokens detected: {', '.join(sorted(blocked))}")
    if strict_lineage and unknown_lineage:
        raise ValueError(f"Leakage policy violation: missing lineage for features: {', '.join(sorted(unknown_lineage))}")

    return {
        "status": "pass",
        "blocked_count": 0,
        "unknown_lineage_count": len(unknown_lineage),
        "unknown_lineage_features": sorted(unknown_lineage),
        "lineage_feature_count": len(lineage),
    }
