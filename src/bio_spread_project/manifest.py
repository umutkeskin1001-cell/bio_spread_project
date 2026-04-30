"""Manifest helpers for BioSpread artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bio_spread_project.input_selection import InputSelection
from bio_spread_project.runtime_policy import EnforcementPolicy


def portable_path(path: Path, *, root: Path | None = None) -> str:
    resolved = path.resolve()
    anchor = (root or Path(__file__).resolve().parents[2]).resolve()
    try:
        return str(resolved.relative_to(anchor))
    except ValueError:
        return str(resolved)


def build_manifest(
    *,
    selection: InputSelection,
    run_mode: str,
    policy: EnforcementPolicy,
    input_hashes: dict[str, str],
    split_year: int,
    horizon_years: int,
    run_metadata: dict[str, str],
    primary_model: str,
    threshold_sources: dict[str, str],
    quality_gates: dict[str, bool],
    artifacts: dict[str, str],
    environment: dict[str, Any] | None = None,
    semantic_input_hashes: dict[str, str] | None = None,
    source_fingerprint: str = "unknown",
    config_fingerprint: str = "unknown",
    dependency_fingerprint: str = "unknown",
) -> dict[str, Any]:
    inputs = {
        name: portable_path(path)
        for name, path in (
            ("input", selection.candidate_inputs.get("input")),
            ("records", selection.resolved_records_path),
            ("amr", selection.resolved_amr_path),
            ("geo_spread_features", selection.candidate_inputs.get("geo_spread_features")),
        )
        if path is not None and name in input_hashes
    }
    candidates = {
        name: portable_path(path)
        for name, path in selection.candidate_inputs.items()
        if path is not None
    }
    return {
        "project": "BioSpread",
        "run_mode": run_mode,
        "input_mode": selection.input_mode,
        "inputs": inputs,
        "input_candidates": candidates,
        "selection_reason": selection.selection_reason,
        "input_hashes": input_hashes,
        "semantic_input_hashes": semantic_input_hashes or {},
        "split_year": split_year,
        "horizon_years": horizon_years,
        "run_id": run_metadata["run_id"],
        "created_at_utc": run_metadata["created_at_utc"],
        "primary_model": primary_model,
        "policy": {
            "fail_on_quality_gates": policy.fail_on_quality_gates,
            "fail_on_drift_fail": policy.fail_on_drift_fail,
            "fail_on_trend_fail": policy.fail_on_trend_fail,
            "require_trend_evidence": policy.require_trend_evidence,
            "require_explicit_surface": policy.require_explicit_surface,
        },
        "threshold_sources": threshold_sources,
        "quality_gates": quality_gates,
        "artifacts": artifacts,
        "environment": environment or {},
        "git_commit": (environment or {}).get("git_commit", "unknown"),
        "source_fingerprint": source_fingerprint,
        "config_fingerprint": config_fingerprint,
        "dependency_fingerprint": dependency_fingerprint,
        "random_seed_policy": {
            "model_random_state": 42,
            "cv_random_state": 7,
            "bootstrap_seed": 7,
            "permutation_importance_seed": 7,
        },
    }
