from pathlib import Path

from fastapi.testclient import TestClient

import dna_sentinel.api as api
from dna_sentinel.api import app
from dna_sentinel.model import KmerTransformer, KmerTransformerConfig
from dna_sentinel.utils import InferenceService


def test_inference_service_supports_kmer_transformer(tmp_path: Path):
    kt_cfg = KmerTransformerConfig(hidden_dim=16, n_heads=2, n_layers=1, n_kmer_features=128)
    kt_model = KmerTransformer(kt_cfg)
    kt_path = tmp_path / "kmer_transformer.pt"
    kt_model.save(kt_path)

    kt_service = InferenceService(str(kt_path))
    kt_pred = kt_service.predict("test_query", "ATGCGT" * 10)
    assert kt_pred["sequence_id"] == "test_query"
    assert "risk_score" in kt_pred
    assert "mobility_probs" in kt_pred
    assert len(kt_pred["mobility_probs"]) == 3
    assert isinstance(kt_pred["top_windows"], list)


def test_api_endpoints_with_mocked_service(tmp_path: Path, monkeypatch):
    class MockService:
        def predict(self, sequence_id: str, dna: str) -> dict:
            return {
                "sequence_id": sequence_id,
                "mobility_probs": [0.1, 0.2, 0.7],
                "amr_probability": 0.8,
                "expansion_probability": 0.9,
                "risk_score": 0.85,
                "top_windows": [{"start": 0.0, "end": 100.0, "weight": 0.9}],
            }

    with TestClient(app) as client:
        monkeypatch.setattr(api, "service", MockService())

        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

        response = client.post(
            "/predict",
            json={"sequence_id": "test_api", "dna": "ATGCGT" * 20},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["sequence_id"] == "test_api"
        assert data["risk_score"] == 0.85
        assert data["mobility_probs"] == [0.1, 0.2, 0.7]

        response = client.post(
            "/predict-batch",
            json={
                "sequences": [
                    {"sequence_id": "seq1", "dna": "ATGCGT" * 20},
                    {"sequence_id": "seq2", "dna": "ATGCGT" * 25},
                ]
            },
        )
        assert response.status_code == 200
        batch_data = response.json()
        assert len(batch_data) == 2
        assert batch_data[0]["sequence_id"] == "seq1"
        assert batch_data[1]["sequence_id"] == "seq2"
