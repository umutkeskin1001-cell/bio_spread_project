from pathlib import Path

from fastapi.testclient import TestClient

import dna_sentinel.api as api
from dna_sentinel.api import app
from dna_sentinel.model import Cassiopeia, CassiopeiaConfig
from dna_sentinel.utils import InferenceService, predict_one, revcomp


def test_inference_service_supports_cassiopeia(tmp_path: Path):
    model = Cassiopeia()
    path = tmp_path / "cassiopeia.pt"
    model.save(path)
    svc = InferenceService(str(path))
    pred = svc.predict("test_query", "ATGCGT" * 10)
    assert pred["sequence_id"] == "test_query" and "risk_score" in pred and len(pred["mobility_probs"]) == 3
    preds = svc.predict_batch([("q1", "ATGCGT" * 10), ("q2", "ATGCGT" * 12)])
    assert len(preds) == 2


def test_predict_one_respects_model_window_budget():
    pred = predict_one(Cassiopeia(CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, n_layers=1, max_windows=56)), "wide", "ACGT" * 200)
    assert pred.sequence_id == "wide" and len(pred.mobility_probs) == 3


def test_predict_one_is_revcomp_invariant():
    model = Cassiopeia(CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, n_layers=1, max_windows=56))
    dna = "ATGCGT" * 200
    a = predict_one(model, "query", dna)
    b = predict_one(model, "query", revcomp(dna))
    assert a.mobility_probs == b.mobility_probs and a.amr_probability == b.amr_probability


def test_api_endpoints_with_mocked_service(tmp_path, monkeypatch):
    class MockService:
        def predict(self, sequence_id, dna):
            return {"sequence_id": sequence_id, "mobility_probs": [0.1, 0.2, 0.7], "amr_probability": 0.8, "expansion_probability": 0.9, "risk_score": 0.85, "top_windows": [{"window": 0.0, "weight": 0.9}]}
        def predict_batch(self, sequences):
            return [self.predict(s, d) for s, d in sequences]

    monkeypatch.setenv("CASSIOPEIA_CHECKPOINT", str(tmp_path / "nonexistent.pt"))
    with TestClient(app) as client:
        monkeypatch.setattr(api, "service", MockService())
        assert client.get("/health").status_code == 200
        r = client.post("/predict", json={"sequence_id": "test", "dna": "ATGCGT" * 20})
        assert r.status_code == 200 and r.json()["risk_score"] == 0.85
        r = client.post("/predict-batch", json={"sequences": [{"sequence_id": "s1", "dna": "ATGCGT" * 20}, {"sequence_id": "s2", "dna": "ATGCGT" * 25}]})
        assert r.status_code == 200 and len(r.json()) == 2


def test_api_validates_dna_length(tmp_path, monkeypatch):
    monkeypatch.setenv("CASSIOPEIA_CHECKPOINT", str(tmp_path / "nonexistent.pt"))
    with TestClient(app) as client:
        r = client.post("/predict", json={"sequence_id": "test", "dna": ""})
        assert r.status_code == 422


def test_api_health_contains_uptime(tmp_path, monkeypatch):
    monkeypatch.setenv("CASSIOPEIA_CHECKPOINT", str(tmp_path / "nonexistent.pt"))
    with TestClient(app) as client:
        r = client.get("/health")
        assert "uptime_seconds" in r.json()
