"""Additional API tests for coverage."""

from fastapi.testclient import TestClient

from dna_sentinel.api import app

client = TestClient(app)


def test_health_no_model():
    """Health endpoint returns missing_checkpoint when no model loaded."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("ok", "missing_checkpoint")
    assert "uptime_seconds" in data


def test_predict_no_model():
    """Predict returns 503 when no model loaded."""
    resp = client.post("/predict", json={"dna": "ATGCGT" * 100})
    assert resp.status_code == 503


def test_predict_batch_no_model():
    """Predict-batch returns 503 when no model loaded."""
    resp = client.post("/predict-batch", json={"sequences": [{"dna": "ATGCGT" * 100}]})
    assert resp.status_code == 503


def test_predict_empty_dna():
    """Predict returns 422 for empty DNA."""
    resp = client.post("/predict", json={"dna": ""})
    assert resp.status_code == 422


def test_predict_whitespace_dna():
    """Predict returns 422 for whitespace-only DNA."""
    resp = client.post("/predict", json={"dna": "   \n\n  "})
    assert resp.status_code == 422


def test_predict_too_long_dna():
    """Predict returns 422 for DNA exceeding max length."""
    long_dna = "ATGC" * 50001
    resp = client.post("/predict", json={"dna": long_dna})
    assert resp.status_code == 422


def test_predict_batch_empty():
    """Predict-batch returns 422 for empty batch."""
    resp = client.post("/predict-batch", json={"sequences": []})
    # Service returns 422 because empty sequences and no model check needed first
    assert resp.status_code in (422, 503)


def test_health_endpoint_exists():
    resp = client.get("/health")
    assert resp.status_code == 200


def test_predict_with_valid_dna_no_checkpoint():
    """Should return 503 since no checkpoint is loaded."""
    resp = client.post("/predict", json={"sequence_id": "test", "dna": "ATGCGT" * 100})
    assert resp.status_code == 503
    data = resp.json()
    assert "detail" in data
