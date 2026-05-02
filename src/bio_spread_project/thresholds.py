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
    return ThresholdBundle(
        quality=load_quality_thresholds(quality_path),
        drift=load_drift_thresholds(drift_path),
        trend=load_trend_thresholds(trend_path),
    )


def thresholds_to_dict(bundle: ThresholdBundle) -> dict[str, Any]:
    return {
        "quality": bundle.quality.__dict__,
        "drift": bundle.drift.__dict__,
        "trend": bundle.trend.__dict__,
    }
