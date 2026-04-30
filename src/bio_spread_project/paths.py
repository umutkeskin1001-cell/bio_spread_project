from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    data_root: Path

    @classmethod
    def from_env(cls) -> "ProjectPaths":
        root = project_root()
        data = os.environ.get("BIO_SPREAD_DATA_ROOT")
        return cls(project_root=root, data_root=Path(data or root / "data").expanduser().resolve())

    @property
    def raw_backbones(self) -> Path:
        return self.data_root / "raw" / "plasmid_backbones.tsv"

    @property
    def raw_amr(self) -> Path:
        return self.data_root / "raw" / "amr.tsv"

    @property
    def geo_spread_features(self) -> Path:
        return self.data_root / "project_inputs" / "geo_spread" / "inputs" / "backbone_scored.tsv"

    @property
    def default_output_dir(self) -> Path:
        return self.project_root / "reports" / "run"


def default_raw_backbones_path() -> Path:
    return ProjectPaths.from_env().raw_backbones


def default_raw_amr_path() -> Path:
    return ProjectPaths.from_env().raw_amr


def default_geo_spread_features_path() -> Path:
    return ProjectPaths.from_env().geo_spread_features


def default_output_dir() -> Path:
    return ProjectPaths.from_env().default_output_dir


# Backward-compatible constants for legacy imports.
PROJECT_ROOT = project_root()
DEFAULT_RAW_BACKBONES_PATH = default_raw_backbones_path()
DEFAULT_RAW_AMR_PATH = default_raw_amr_path()
DEFAULT_GEO_SPREAD_FEATURES_PATH = default_geo_spread_features_path()
DEFAULT_OUTPUT_DIR = default_output_dir()
