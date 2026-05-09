from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from bio_spread_project.validation import to_int


@dataclass(frozen=True)
class EnforcementPolicy:
    fail_on_quality_gates: bool = False
    fail_on_drift_fail: bool = False
    fail_on_trend_fail: bool = False
    require_trend_evidence: bool = False
    require_explicit_surface: bool = False
    require_strict_lineage: bool = False

    @property
    def allow_conditional_release(self) -> bool:
        return not self.require_trend_evidence


@dataclass(frozen=True)
class PipelineConfig:
    input_path: Path | None = None
    backbone_records_path: Path | None = None
    amr_path: Path | None = None
    geo_spread_features_path: Path | None = None
    external_holdout_path: Path | None = None
    baseline_benchmark_path: Path | None = None
    drift_thresholds_path: Path | None = None
    trend_thresholds_path: Path | None = None
    quality_thresholds_path: Path | None = None
    output_dir: Path = field(default_factory=lambda: Path("reports/latest"))
    run_mode: str = "auto"
    split_year: int = 2020
    horizon_years: int = 3
    policy: EnforcementPolicy = field(default_factory=EnforcementPolicy)
    triage_budget: int | None = None

    def __post_init__(self) -> None:
        accepted_modes = {"auto", "raw", "geo", "input", "active_radar"}
        if self.run_mode not in accepted_modes:
            raise ValueError(f"run_mode must be one of {sorted(accepted_modes)}, got {self.run_mode!r}")
        if self.horizon_years <= 0:
            raise ValueError(f"horizon_years must be > 0, got {self.horizon_years}")
        if not (1900 <= int(self.split_year) <= 2100):
            raise ValueError(f"split_year outside accepted range [1900, 2100]: {self.split_year}")
        if self.input_path is not None and self.backbone_records_path is not None:
            raise ValueError("Provide either input_path or backbone_records_path, not both")

        object.__setattr__(self, "output_dir", Path(self.output_dir))
        for field_name in (
            "input_path",
            "backbone_records_path",
            "amr_path",
            "geo_spread_features_path",
            "external_holdout_path",
            "baseline_benchmark_path",
            "drift_thresholds_path",
            "trend_thresholds_path",
            "quality_thresholds_path",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, Path(value))
        object.__setattr__(self, "split_year", to_int("split_year", self.split_year))
        object.__setattr__(self, "horizon_years", to_int("horizon_years", self.horizon_years))
