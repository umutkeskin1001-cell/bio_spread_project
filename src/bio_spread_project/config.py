"""Configuration for the standalone BioSpread workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    weights: dict[str, float]


@dataclass(frozen=True)
class ProjectConfig:
    models: tuple[ModelSpec, ...]
    max_review_rate: float = 0.80
    calibration_weight: float = 0.25
    discrimination_weight: float = 0.50
    decision_weight: float = 0.25


@dataclass(frozen=True)
class ConfigInventory:
    runtime_files: tuple[str, ...]
    reference_only_files: tuple[str, ...]


def load_project_config() -> ProjectConfig:
    """Return the authoritative built-in model surface for the standalone runtime."""
    return ProjectConfig(
        models=(
            ModelSpec(
                name="mobility",
                description="Mobility-only early warning baseline",
                weights={
                    "mobility": 1.80,
                    "amr": 0.20,
                    "country": 0.25,
                    "host": 0.10,
                    "clinical": 0.10,
                    "low_knownness": 0.15,
                },
            ),
            ModelSpec(
                name="amr_mobility",
                description="AMR burden plus mobility risk model",
                weights={
                    "mobility": 1.15,
                    "amr": 1.05,
                    "country": 0.70,
                    "host": 0.45,
                    "clinical": 0.25,
                    "low_knownness": 0.20,
                },
            ),
            ModelSpec(
                name="clinical_hybrid",
                description="Clinical-context aware geographic spread model",
                weights={
                    "mobility": 0.90,
                    "amr": 0.95,
                    "country": 0.55,
                    "host": 0.50,
                    "clinical": 0.85,
                    "low_knownness": 0.25,
                },
            ),
        )
    )


def configuration_inventory() -> ConfigInventory:
    root = Path(__file__).resolve().parents[2]
    return ConfigInventory(
        runtime_files=(
            str(root / "project_config" / "quality_thresholds.json"),
            str(root / "project_config" / "drift_thresholds.json"),
            str(root / "project_config" / "trend_thresholds.json"),
            str(root / "project_config" / "baseline_benchmark.json"),
        ),
        reference_only_files=(
            str(root / "project_config" / "config.yaml"),
            str(root / "project_config" / "config" / "benchmarks.yaml"),
            str(root / "project_config" / "config" / "freeze_contract.yaml"),
            str(root / "project_config" / "config" / "import_contract_allowlist.txt"),
            str(root / "project_config" / "config" / "model_compute_tiers.yaml"),
            str(root / "project_config" / "config" / "performance_budgets.yaml"),
            str(root / "project_config" / "config" / "rag_corpus.yaml"),
            str(root / "project_config" / "config" / "research_models.yaml"),
            str(root / "project_config" / "config" / "scoring_weights.yaml"),
        ),
    )
