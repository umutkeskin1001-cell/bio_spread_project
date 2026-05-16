"""Tests for InferenceService."""

import json

import numpy as np
import polars as pl
import pytest
import torch
import yaml

from bio_spread.config.schema import Config
from bio_spread.data.dataset import fit_normalizers, fit_static_normalizers
from bio_spread.models import create_model


@pytest.fixture
def service_artifacts(tmp_path):
    """Create minimal artifacts for InferenceService testing."""
    cfg = Config()

    # Create minimal sequences.tsv
    df = pl.DataFrame(
        {
            "backbone_id": ["A"] * 3,
            "year": [2018, 2019, 2020],
            "n_countries": [1.0, 2.0, 3.0],
            "n_hosts": [1.0, 1.0, 2.0],
            "years_since_first": [0.0, 1.0, 2.0],
            "new_countries_recent": [0.0, 1.0, 1.0],
            "new_countries_2y_ago": [0.0, 0.0, 0.0],
            "n_records": [1.0, 2.0, 3.0],
            "acceleration": [0.0, 1.0, 0.0],
            "niche_breadth": [1.0, 1.0, 0.5],
            "log_size": [8.5] * 3,
            "gc": [0.5] * 3,
            "n_replicon_types": [2.0] * 3,
            "n_relaxase_types": [1.0] * 3,
            "mobility_score": [2.0] * 3,
            "is_conjugative": [1.0] * 3,
            "is_mobilizable": [0.0] * 3,
            "topology": [0.0] * 3,
            "n_orit_types": [2.0] * 3,
            "host_range_rank": [3.0] * 3,
            "hazard_1": [0.0, 1.0, -1.0],
            "hazard_2": [1.0, 1.0, -1.0],
            "hazard_3": [1.0, 1.0, -1.0],
            "n_new_countries": [1.0, 1.0, -1.0],
            "observed": [1.0, 1.0, 0.0],
        }
    )
    df.write_csv(tmp_path / "sequences.tsv", separator="\t")

    # Create split.json
    split = {"train": ["A"], "val": ["A"], "test": ["A"]}
    with open(tmp_path / "split.json", "w") as f:
        json.dump(split, f)

    # Fit and save normalizers
    norm = fit_normalizers(df)
    static_norm = fit_static_normalizers(df)
    np.savez(tmp_path / "normalizers.npz", means=norm[0], stds=norm[1])
    np.savez(tmp_path / "static_normalizers.npz", means=static_norm[0], stds=static_norm[1])

    # Create and save model (no taxonomy to avoid dimension mismatch at inference)
    model = create_model(10, 16, cfg.model)
    model_path = tmp_path / "best_model.pt"
    torch.save(model.state_dict(), model_path)

    # Save config
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump({"model": cfg.model.model_dump()}, f)

    return str(model_path), str(cfg_path), str(tmp_path)


def test_predict_returns_correct_keys(service_artifacts):
    from bio_spread.serving.service import InferenceService

    model_path, cfg_path, feature_dir = service_artifacts
    service = InferenceService(model_path, cfg_path, feature_dir)
    result = service.predict(
        snapshots=[
            {
                "n_countries": 1.0,
                "n_hosts": 1.0,
                "years_since_first": 0.0,
                "new_countries_recent": 0.0,
                "new_countries_2y_ago": 0.0,
                "n_records": 1.0,
                "acceleration": 0.0,
                "niche_breadth": 1.0,
            }
        ],
        static={
            "log_size": 8.5,
            "gc": 0.5,
            "n_replicon_types": 2.0,
            "n_relaxase_types": 1.0,
            "mobility_score": 2.0,
            "is_conjugative": 1.0,
            "is_mobilizable": 0.0,
            "topology": 0.0,
            "n_orit_types": 2.0,
            "host_range_rank": 3.0,
        },
    )
    assert "hazard_year1" in result
    assert "hazard_year2" in result
    assert "hazard_year3" in result
    assert "n_snapshots" in result
    assert 0 <= result["hazard_year1"] <= 1
    assert 0 <= result["hazard_year2"] <= 1
    assert 0 <= result["hazard_year3"] <= 1


def test_predict_empty_snapshots_raises(service_artifacts):
    from bio_spread.serving.service import InferenceService

    model_path, cfg_path, feature_dir = service_artifacts
    service = InferenceService(model_path, cfg_path, feature_dir)
    with pytest.raises(ValueError, match="at least 1"):
        service.predict(
            snapshots=[],
            static={
                "log_size": 8.5,
                "gc": 0.5,
                "n_replicon_types": 2.0,
                "n_relaxase_types": 1.0,
                "mobility_score": 2.0,
                "is_conjugative": 1.0,
                "is_mobilizable": 0.0,
                "topology": 0.0,
                "n_orit_types": 2.0,
                "host_range_rank": 3.0,
            },
        )
