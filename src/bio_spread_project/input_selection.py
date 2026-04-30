"""Input selection helpers for BioSpread pipeline runs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from bio_spread_project.paths import ProjectPaths

RUN_MODES = {"auto", "raw", "geo", "input"}


@dataclass(frozen=True)
class InputSelection:
    source_path: Path
    input_mode: str
    use_geo_reliability: bool
    resolved_records_path: Path | None
    resolved_amr_path: Path | None
    selection_reason: str
    candidate_inputs: dict[str, Path | None]


def ensure_readable_file(path: str | Path, *, label: str, allow_empty: bool = False) -> Path:
    checked = Path(path).expanduser().resolve()
    if not checked.exists() or not checked.is_file():
        raise ValueError(f"{label} path does not exist: {checked}")
    if not os.access(checked, os.R_OK):
        raise ValueError(f"{label} path is not readable: {checked}")
    if not allow_empty and checked.stat().st_size == 0:
        raise ValueError(f"{label} path is empty: {checked}")
    return checked


def select_input_source(
    *,
    run_mode: str,
    input_path: str | Path | None,
    backbone_records_path: str | Path | None,
    amr_path: str | Path | None,
    geo_spread_features_path: str | Path | None,
    require_explicit_surface: bool,
) -> InputSelection:
    if run_mode not in RUN_MODES:
        accepted = ", ".join(sorted(RUN_MODES))
        raise ValueError(f"Unknown run_mode `{run_mode}`; expected one of: {accepted}")
    if input_path is not None and backbone_records_path is not None:
        raise ValueError("Provide either input_path or backbone_records_path, not both")

    packaged_geo_path = (
        Path(geo_spread_features_path)
        if geo_spread_features_path is not None
        else ProjectPaths.from_env().geo_spread_features
    )
    resolved_records_path = Path(backbone_records_path) if backbone_records_path is not None else None
    resolved_amr_path = Path(amr_path) if amr_path is not None else None
    candidate_inputs = {
        "input": Path(input_path) if input_path is not None else None,
        "records": resolved_records_path,
        "amr": resolved_amr_path,
        "geo_spread_features": packaged_geo_path,
    }

    if run_mode == "input":
        if input_path is None:
            raise ValueError("run_mode=`input` requires input_path")
        source_path = ensure_readable_file(input_path, label="input")
        return InputSelection(
            source_path=source_path,
            input_mode="observation_records",
            use_geo_reliability=False,
            resolved_records_path=None,
            resolved_amr_path=resolved_amr_path,
            selection_reason="explicit_input_mode",
            candidate_inputs=candidate_inputs,
        )

    if run_mode == "geo":
        source_path = ensure_readable_file(packaged_geo_path, label="geo_spread_features")
        if resolved_records_path is not None:
            resolved_records_path = ensure_readable_file(resolved_records_path, label="records")
        if resolved_amr_path is not None:
            resolved_amr_path = ensure_readable_file(resolved_amr_path, label="amr")
        return InputSelection(
            source_path=source_path,
            input_mode="geo_reliability_feature_surface",
            use_geo_reliability=True,
            resolved_records_path=resolved_records_path,
            resolved_amr_path=resolved_amr_path,
            selection_reason="explicit_geo_mode",
            candidate_inputs=candidate_inputs,
        )

    if run_mode == "raw":
        if resolved_records_path is None:
            raise ValueError("run_mode=`raw` requires backbone_records_path")
        source_path = ensure_readable_file(resolved_records_path, label="records")
        if resolved_amr_path is not None:
            resolved_amr_path = ensure_readable_file(resolved_amr_path, label="amr")
        return InputSelection(
            source_path=source_path,
            input_mode="raw_backbone_records",
            use_geo_reliability=False,
            resolved_records_path=source_path,
            resolved_amr_path=resolved_amr_path,
            selection_reason="explicit_raw_mode",
            candidate_inputs=candidate_inputs,
        )

    if input_path is not None:
        source_path = ensure_readable_file(input_path, label="input")
        return InputSelection(
            source_path=source_path,
            input_mode="observation_records",
            use_geo_reliability=False,
            resolved_records_path=None,
            resolved_amr_path=resolved_amr_path,
            selection_reason="auto_selected_input_csv",
            candidate_inputs=candidate_inputs,
        )
    if resolved_records_path is None:
        raise ValueError("Provide backbone_records_path or input_path")

    resolved_records_path = ensure_readable_file(resolved_records_path, label="records")
    if resolved_amr_path is not None:
        resolved_amr_path = ensure_readable_file(resolved_amr_path, label="amr")

    if packaged_geo_path.exists():
        if require_explicit_surface:
            raise ValueError(
                "auto mode detected a GeoSpread feature surface and would promote the run to geo mode; "
                "re-run with --mode geo or disable --require-explicit-surface"
            )
        return InputSelection(
            source_path=packaged_geo_path,
            input_mode="geo_reliability_feature_surface",
            use_geo_reliability=True,
            resolved_records_path=resolved_records_path,
            resolved_amr_path=resolved_amr_path,
            selection_reason="auto_selected_geo_surface_because_feature_surface_exists",
            candidate_inputs=candidate_inputs,
        )
    return InputSelection(
        source_path=resolved_records_path,
        input_mode="raw_backbone_records",
        use_geo_reliability=False,
        resolved_records_path=resolved_records_path,
        resolved_amr_path=resolved_amr_path,
        selection_reason="auto_selected_raw_records_because_geo_surface_is_absent",
        candidate_inputs=candidate_inputs,
    )
