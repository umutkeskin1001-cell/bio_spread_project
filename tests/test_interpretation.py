"""Tests for biological interpretation module."""

import numpy as np

from dna_sentinel.utils import (
    _confidence_label,
    compute_risk_score,
    interpret_amr,
    interpret_expansion,
    interpret_mobility,
    interpret_prediction,
)


def test_confidence_label_high():
    assert _confidence_label(0.85) == "HIGH"
    assert _confidence_label(0.80) == "HIGH"


def test_confidence_label_medium():
    assert _confidence_label(0.70) == "MEDIUM"
    assert _confidence_label(0.60) == "MEDIUM"


def test_confidence_label_low():
    assert _confidence_label(0.30) == "LOW"
    assert _confidence_label(0.59) == "LOW"


def test_interpret_mobility_non_mobilizable():
    result = interpret_mobility([0.85, 0.10, 0.05])
    assert result["label"] == "non-mobilizable"
    assert result["confidence"] == "HIGH"
    assert "class_probabilities" in result


def test_interpret_mobility_mobilizable():
    result = interpret_mobility([0.20, 0.65, 0.15])
    assert result["label"] == "mobilizable"
    assert result["confidence"] == "MEDIUM"


def test_interpret_mobility_conjugative():
    result = interpret_mobility([0.05, 0.15, 0.80])
    assert result["label"] == "conjugative"
    assert result["confidence"] == "HIGH"


def test_interpret_mobility_empty():
    result = interpret_mobility([])
    assert result["label"] == "unknown"


def test_interpret_mobility_short():
    result = interpret_mobility([0.5])
    assert result["label"] == "unknown"


def test_interpret_amr_high():
    result = interpret_amr(0.90)
    assert result["confidence"] == "HIGH"
    assert isinstance(result["matched_card_families"], list)
    assert result["note"] != "unknown"


def test_interpret_amr_low():
    result = interpret_amr(0.30)
    assert result["confidence"] == "LOW"
    assert "no match could be made" in result["note"]


def test_interpret_amr_medium():
    result = interpret_amr(0.70)
    assert result["confidence"] == "MEDIUM"


def test_interpret_expansion_conjugative_amr_high():
    result = interpret_expansion(0.85, "conjugative", "HIGH")
    assert result["confidence"] in ("HIGH", "MEDIUM")
    assert "co-occurrence" in result["reasoning"]


def test_interpret_expansion_non_mobilizable():
    result = interpret_expansion(0.30, "non-mobilizable", "LOW")
    assert "non-mobilizable" in result["reasoning"]


def test_interpret_expansion_mobilizable():
    result = interpret_expansion(0.50, "mobilizable", "MEDIUM")
    assert "mobilizable" in result["reasoning"]


def test_interpret_prediction_full():
    pred = {
        "mobility_probs": [0.10, 0.20, 0.70],
        "amr_probability": 0.85,
        "expansion_probability": 0.75,
    }
    result = interpret_prediction(pred)
    assert "mobility" in result
    assert "amr" in result
    assert "expansion" in result
    assert "overall_risk_score" in result
    assert result["mobility"]["label"] == "conjugative"
    assert result["amr"]["confidence"] == "HIGH"
    assert result["disclaimer"] is not None
    assert "kullanılamaz" in result["disclaimer"]


def test_interpret_prediction_low_confidence():
    pred = {
        "mobility_probs": [0.50, 0.30, 0.20],
        "amr_probability": 0.35,
        "expansion_probability": 0.40,
    }
    result = interpret_prediction(pred)
    assert result["mobility"]["label"] == "non-mobilizable"
    assert result["amr"]["confidence"] == "LOW"


def test_compute_risk_score_default_weights():
    score = compute_risk_score([0.7, 0.2, 0.1], 0.5, 0.3)
    expected = 0.4 * 0.3 + 0.3 * 0.5 + 0.3 * 0.3
    assert abs(score - expected) < 1e-6


def test_compute_risk_score_mobile():
    score = compute_risk_score([0.1, 0.3, 0.6], 0.9, 0.8, (0.4, 0.3, 0.3))
    expected = 0.4 * 0.9 + 0.3 * 0.9 + 0.3 * 0.8
    assert abs(score - expected) < 1e-6


def test_compute_risk_score_zero():
    score = compute_risk_score([1.0, 0.0, 0.0], 0.0, 0.0)
    assert score == 0.0


def test_interpret_prediction_disclaimer_present():
    pred = {"mobility_probs": [0.8, 0.15, 0.05], "amr_probability": 0.1, "expansion_probability": 0.2}
    result = interpret_prediction(pred)
    assert "disclaimer" in result
    assert isinstance(result["disclaimer"], str)
    assert len(result["disclaimer"]) > 10


def test_interpret_amr_with_matched_families():
    result = interpret_amr(0.90)
    assert isinstance(result.get("matched_card_families"), list)
    if result["matched_card_families"]:
        assert all(isinstance(f, str) for f in result["matched_card_families"])
