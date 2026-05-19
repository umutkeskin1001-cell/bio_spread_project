from pathlib import Path

from fastapi.testclient import TestClient

import dna_sentinel.api as api
from dna_sentinel.api import app
from dna_sentinel.dataset import DnaDataset, LabeledSequence
from dna_sentinel.kmer import KmerConfig, KmerSentinel
from dna_sentinel.model import DnaSentinel, DnaSentinelConfig
from dna_sentinel.service import InferenceService
from dna_sentinel.train import TrainConfig, train_model


def test_inference_service_supports_all_formats(tmp_path: Path):
    records = [
        LabeledSequence("p1", "ATGCGT" * 20, 2, 1, 1),
        LabeledSequence("p2", "ATGCGT" * 18, 2, 1, 1),
        LabeledSequence("n1", "TTAACC" * 20, 0, 0, 0),
        LabeledSequence("n2", "TTAACC" * 18, 0, 0, 0),
        LabeledSequence("m1", "CCCCGG" * 20, 1, 0, 0),
        LabeledSequence("m2", "GGCCCC" * 20, 1, 0, 0),
    ]

    kmer_path = tmp_path / "kmer.joblib"
    kmer_model = KmerSentinel.train(records, KmerConfig(n_features=1024, max_iter=200))
    kmer_model.save(kmer_path)

    ds = DnaDataset(records, window_size=48, stride=24, max_windows=4)
    neural_model = DnaSentinel(
        DnaSentinelConfig(channels=16, layers=2, window_size=48, stride=24, max_windows=4)
    )
    neural_path, _ = train_model(
        neural_model, ds, ds, TrainConfig(epochs=1, batch_size=3, artifact_dir=tmp_path, seed=5)
    )

    kmer_service = InferenceService(kmer_path)
    kmer_pred = kmer_service.predict("test_query", "ATGCGT" * 10)
    assert kmer_pred["sequence_id"] == "test_query"
    assert "risk_score" in kmer_pred
    assert "mobility_probs" in kmer_pred
    assert len(kmer_pred["mobility_probs"]) == 3
    assert isinstance(kmer_pred["top_windows"], list)

    neural_service = InferenceService(neural_path)
    neural_pred = neural_service.predict("test_query", "ATGCGT" * 10)
    assert neural_pred["sequence_id"] == "test_query"
    assert "risk_score" in neural_pred
    assert "mobility_probs" in neural_pred
    assert len(neural_pred["mobility_probs"]) == 3
    assert isinstance(neural_pred["top_windows"], list)

    from dna_sentinel.kmer_transformer import KmerTransformer, KmerTransformerConfig
    kt_cfg = KmerTransformerConfig(hidden_dim=16, n_heads=2, n_layers=1, n_kmer_features=128)
    kt_model = KmerTransformer(kt_cfg)
    kt_path = tmp_path / "kmer_transformer.pt"
    kt_model.save(kt_path)

    kt_service = InferenceService(kt_path)
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
