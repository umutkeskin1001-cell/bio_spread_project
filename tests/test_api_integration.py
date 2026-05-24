from pathlib import Path

from fastapi.testclient import TestClient

import dna_sentinel.api as api
from dna_sentinel.api import app
from dna_sentinel.model import Cassiopeia
from dna_sentinel.utils import InferenceService


def test_inference_service_supports_cassiopeia(tmp_path: Path):
    model = Cassiopeia()
    path = tmp_path / "cassiopeia.pt"
    model.save(path)
    service = InferenceService(str(path))
    pred = service.predict("test_query", "ATGCGT" * 10)
    assert pred["sequence_id"] == "test_query"
    assert "risk_score" in pred
    assert "mobility_probs" in pred
    assert len(pred["mobility_probs"]) == 3
    assert isinstance(pred["top_windows"], list)
    preds = service.predict_batch([("q1", "ATGCGT" * 10), ("q2", "ATGCGT" * 12)])
    assert len(preds) == 2


def test_api_endpoints_with_mocked_service(tmp_path: Path, monkeypatch):
    class MockService:
        def predict(self, sequence_id, dna):
            return {
            "sequence_id": sequence_id,
            "mobility_probs": [0.1, 0.2, 0.7],
            "amr_probability": 0.8,
            "expansion_probability": 0.9,
            "risk_score": 0.85,
            "top_windows": [{"window": 0.0, "weight": 0.9}],
        }
        def predict_batch(self, sequences):
            parsed = [(s.sequence_id, s.dna) for s in sequences]
            return [self.predict(sid, dna) for sid, dna in parsed]

    monkeypatch.setenv("CASSIOPEIA_CHECKPOINT", str(tmp_path / "nonexistent.pt"))
    with TestClient(app) as client:
        monkeypatch.setattr(api, "service", MockService())
        r = client.get("/health")
        assert r.status_code == 200 and r.json()["status"] == "ok"
        r = client.post("/predict", json={"sequence_id": "test_api", "dna": "ATGCGT" * 20})
        assert r.status_code == 200 and r.json()["risk_score"] == 0.85
        seqs = [{"sequence_id": "s1", "dna": "ATGCGT" * 20},
                {"sequence_id": "s2", "dna": "ATGCGT" * 25}]
        r = client.post("/predict-batch", json={"sequences": seqs})
        assert r.status_code == 200 and len(r.json()) == 2
