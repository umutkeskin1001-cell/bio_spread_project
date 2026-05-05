from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_data_root() -> Path:
    configured = os.environ.get("BIO_SPREAD_DATA_ROOT")
    return Path(configured or (project_root() / "data")).expanduser().resolve()


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    data_root: Path

    @classmethod
    def from_env(cls) -> "ProjectPaths":
        return cls(project_root=project_root(), data_root=resolve_data_root())

    @property
    def raw_backbones(self) -> Path:
        # SOTA default: curated backbone table built from external-source pipeline
        # (NCBI Pathogen Detection + PLSDB + PubMLST harmonization layer).
        return self.data_root / "project_inputs" / "silver" / "plasmid_backbones.tsv"

    @property
    def raw_amr(self) -> Path:
        # SOTA default: curated AMR table from external-source ingestion layer.
        return self.data_root / "project_inputs" / "raw" / "amr.tsv"

    @property
    def geo_spread_features(self) -> Path:
        # SOTA default: strict train-only surface to avoid identity overlap with external holdout.
        return self.data_root / "project_inputs" / "geo_spread" / "inputs" / "backbone_scored_train_only.tsv"

    @property
    def external_holdout_geo_spread_features(self) -> Path:
        return self.data_root / "project_inputs" / "geo_spread" / "inputs" / "external_holdout_curated_v1.tsv"

    @property
    def default_output_dir(self) -> Path:
        return self.project_root / "reports" / "latest"


@dataclass(frozen=True)
class FeatureWeights:
    mobility: float
    amr: float
    country: float
    host: float
    clinical: float
    low_knownness: float

    def keys(self) -> tuple[str, ...]:
        return ("mobility", "amr", "country", "host", "clinical", "low_knownness")

    def __getitem__(self, key: str) -> float:
        return float(getattr(self, key))

    def items(self) -> tuple[tuple[str, float], ...]:
        return tuple((key, self[key]) for key in self.keys())


@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    weights: FeatureWeights


@dataclass(frozen=True)
class ProjectConfig:
    models: tuple[ModelSpec, ...]


def load_project_config() -> ProjectConfig:
    return ProjectConfig(
        models=(
            ModelSpec(
                name="mobility",
                description="Mobility-only early warning baseline",
                weights=FeatureWeights(1.80, 0.20, 0.25, 0.10, 0.10, 0.15),
            ),
            ModelSpec(
                name="amr_mobility",
                description="AMR burden plus mobility risk model",
                weights=FeatureWeights(1.15, 1.05, 0.70, 0.45, 0.25, 0.20),
            ),
            ModelSpec(
                name="clinical_hybrid",
                description="Clinical-context aware geographic spread model",
                weights=FeatureWeights(0.90, 0.95, 0.55, 0.50, 0.85, 0.25),
            ),
        )
    )
