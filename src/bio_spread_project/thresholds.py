from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bio_spread_project.governance import (
    DriftThresholds,
    QualityThresholds,
    TrendThresholds,
    load_drift_thresholds,
    load_quality_thresholds,
    load_trend_thresholds,
)


@dataclass(frozen=True)
class ThresholdBundle:
    quality: QualityThresholds
    drift: DriftThresholds
    trend: TrendThresholds


def load_thresholds(
    *,
    quality_path: str | Path | None = None,
    drift_path: str | Path | None = None,
    trend_path: str | Path | None = None,
) -> ThresholdBundle:
    from bio_spread_project.config_loader import ProjectPaths

    paths = ProjectPaths.from_env()
    root = paths.project_root
    
    q = quality_path or (root / "project_config" / "quality_thresholds.json")
    d = drift_path or (root / "project_config" / "drift_thresholds.json")
    t = trend_path or (root / "project_config" / "trend_thresholds.json")

    return ThresholdBundle(
        quality=load_quality_thresholds(q if Path(q).exists() else None),
        drift=load_drift_thresholds(d if Path(d).exists() else None),
        trend=load_trend_thresholds(t if Path(t).exists() else None),
    )


def thresholds_to_dict(bundle: ThresholdBundle) -> dict[str, Any]:
    return {
        "quality": bundle.quality.__dict__,
        "drift": bundle.drift.__dict__,
        "trend": bundle.trend.__dict__,
    }
