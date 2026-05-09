from __future__ import annotations

from bio_spread_project.geo_reliability import FEATURE_COLUMNS, GeoBioReliabilityModel

# Backward-compatible public alias kept for older imports.
GeoSpreadEnsemble = GeoBioReliabilityModel


__all__ = ["GeoSpreadEnsemble", "FEATURE_COLUMNS"]
