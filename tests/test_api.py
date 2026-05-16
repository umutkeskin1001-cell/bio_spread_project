"""Tests for API endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_service():
    service = MagicMock()
    service.predict.return_value = {
        "hazard_year1": 0.5,
        "hazard_year2": 0.7,
        "hazard_year3": 0.3,
        "n_snapshots": 3,
    }
    return service


def test_health_endpoint():
    with patch("api.service", None):
        from api import app

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "loading")


def test_predict_valid_request(mock_service):
    with patch("api.service", mock_service):
        from api import app

        client = TestClient(app)
        resp = client.post(
            "/predict",
            json={
                "snapshots": [
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
                "static": {
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
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "hazard_year1" in data


def test_predict_empty_snapshots(mock_service):
    with patch("api.service", mock_service):
        from api import app

        client = TestClient(app)
        resp = client.post(
            "/predict",
            json={
                "snapshots": [],
                "static": {
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
            },
        )
        assert resp.status_code == 422  # validation error
