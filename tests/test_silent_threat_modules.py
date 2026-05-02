from __future__ import annotations

import numpy as np
import polars as pl
import torch

from bio_spread_project.conformal import ConformalPredictor
from bio_spread_project.gated_fusion import KnownnessGatedFusion
from bio_spread_project.grps import compute_grps
from bio_spread_project.losses import focal_pairwise_loss, reliability_weighted_propensity


class _DummyModel:
    def predict_proba(self, X):
        X = np.asarray(X)
        p = 1.0 / (1.0 + np.exp(-X[:, 0]))
        return np.stack([1 - p, p], axis=1)


def test_compute_grps_returns_scores_for_low_knownness_backbones() -> None:
    emb = pl.DataFrame(
        {
            "backbone_id": ["a", "b", "c"],
            "esm2_embed_0": [1.0, 0.9, -1.0],
            "esm2_embed_1": [0.0, 0.1, 0.0],
        }
    )
    labels = pl.DataFrame({"backbone_id": ["a", "b"], "label_geo_spread": [1, 0]})
    out = compute_grps(["c"], emb, labels, ["esm2_embed_0", "esm2_embed_1"], n_neighbors=2)
    assert out.height == 1
    assert 0.0 <= float(out["grps"][0]) <= 1.0


def test_gated_fusion_output_shape() -> None:
    module = KnownnessGatedFusion(hist_dim=4, intrin_dim=3, latent_dim=5)
    hist = torch.randn(6, 4)
    intrin = torch.randn(6, 3)
    knownness = torch.rand(6, 1)
    out = module(hist, intrin, knownness)
    assert out.shape == (6, 5)


def test_reliability_weighted_propensity_and_weighted_focal_loss() -> None:
    weights = reliability_weighted_propensity(
        {
            "n_records_pre": [1, 2, 3, 4],
            "metadata_support_depth_norm": [0.6, 0.2, 0.9, 0.1],
            "assignment_confidence_norm": [0.8, 0.3, 0.7, 0.4],
            "backbone_purity_norm": [0.9, 0.2, 0.8, 0.2],
        }
    )
    scores = torch.tensor([2.0, 1.0, -1.0, -2.0])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    knownness = torch.tensor([0.8, 0.7, 0.3, 0.2])
    loss = focal_pairwise_loss(scores, labels, knownness, weights=weights)
    assert torch.isfinite(loss)
    assert float(loss) >= 0.0


def test_conformal_predictor_binary_set_output() -> None:
    model = _DummyModel()
    calib_X = np.array([[0.1], [0.2], [0.3], [0.4]])
    calib_y = np.array([0, 0, 1, 1])
    cp = ConformalPredictor(model, calib_X, calib_y, alpha=0.1)
    out = cp.predict_with_set(np.array([[0.1], [1.0]]))
    assert "risk_probability" in out
    assert "lower" in out and "upper" in out
    assert len(out["risk_probability"]) == 2
