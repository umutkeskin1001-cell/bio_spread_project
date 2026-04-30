from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union, cast

import polars as pl

from bio_spread_project.io_utils import write_json
from bio_spread_project.validation import require_range, to_float


@dataclass(frozen=True)
class TrendThresholds:
    roc_auc_max_drop: float = 0.02
    average_precision_max_drop: float = 0.03
    min_gate_pass_rate: float = 0.90
    required_entries_for_go: int = 20


def load_trend_thresholds(path: str | Path | None = None) -> TrendThresholds:
    if path is None:
        return TrendThresholds()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return TrendThresholds(
        roc_auc_max_drop=require_range("roc_auc_max_drop", to_float("roc_auc_max_drop", payload.get("roc_auc_max_drop", 0.02)), minimum=0.0, maximum=1.0),
        average_precision_max_drop=require_range("average_precision_max_drop", to_float("average_precision_max_drop", payload.get("average_precision_max_drop", 0.03)), minimum=0.0, maximum=1.0),
        min_gate_pass_rate=require_range("min_gate_pass_rate", to_float("min_gate_pass_rate", payload.get("min_gate_pass_rate", 0.90)), minimum=0.0, maximum=1.0),
        required_entries_for_go=int(require_range("required_entries_for_go", to_float("required_entries_for_go", payload.get("required_entries_for_go", 20)), minimum=2.0, maximum=1000000.0)),
    )


def load_model_registry(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pl.DataFrame()
    return pl.read_ndjson(path)


def evaluate_model_registry_trend(
    *,
    entries: pl.DataFrame | list[dict[str, Any]],
    window_size: int,
    thresholds: TrendThresholds,
) -> dict[str, Any]:
    df = entries if isinstance(entries, pl.DataFrame) else pl.DataFrame(entries)
    resolved_window = max(2, int(window_size))
    required_entries = max(2 * resolved_window, thresholds.required_entries_for_go)

    if df.is_empty() or len(df) < required_entries:
        return {
            "status": "insufficient_data", "all_passed": True, "trend_evidence_sufficient": False,
            "required_entries": required_entries, "available_entries": len(df), "window_size": resolved_window,
        }

    # Slice indices
    recent = df.tail(resolved_window)
    previous = df.slice(len(df) - 2 * resolved_window, resolved_window)

    def get_mean(data: pl.DataFrame, col: str) -> float:
        return float(cast(Union[float, int], data[col].mean())) if col in data.columns else 0.0

    recent_auc, prev_auc = get_mean(recent, "roc_auc"), get_mean(previous, "roc_auc")
    recent_ap, prev_ap = get_mean(recent, "average_precision"), get_mean(previous, "average_precision")

    gate_col = "all_quality_gates_passed"
    gate_pass_rate = float(cast(Union[float, int], recent[gate_col].cast(pl.Int32).mean())) if gate_col in recent.columns else 0.0

    auc_delta, ap_delta = recent_auc - prev_auc, recent_ap - prev_ap
    auc_passed = auc_delta >= -thresholds.roc_auc_max_drop
    ap_passed = ap_delta >= -thresholds.average_precision_max_drop
    gate_rate_passed = gate_pass_rate >= thresholds.min_gate_pass_rate

    all_passed = auc_passed and ap_passed and gate_rate_passed
    return {
        "status": "ok", "all_passed": all_passed,
        "trend_evidence_sufficient": True, "window_size": resolved_window,
        "available_entries": len(df), "checks": {
            "roc_auc": {"previous_mean": prev_auc, "recent_mean": recent_auc, "delta": auc_delta, "passed": auc_passed},
            "average_precision": {"previous_mean": prev_ap, "recent_mean": recent_ap, "delta": ap_delta, "passed": ap_passed},
            "gate_pass_rate": {"recent_rate": gate_pass_rate, "min_required": thresholds.min_gate_pass_rate, "passed": gate_rate_passed},
        },
    }


def write_trend_report(path: str | Path, payload: dict[str, Any]) -> Path:
    return write_json(Path(path), payload)
