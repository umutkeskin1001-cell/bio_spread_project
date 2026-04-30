from typing import Any

import numpy as np

from bio_spread_project.model import Prediction


def calibration_summary(predictions: list[Prediction], *, bins: int = 5) -> dict[str, Any]:
    """Return a compact expected calibration error summary using vectorized NumPy logic."""
    if not predictions:
        raise ValueError("Cannot calibrate an empty prediction set")

    probs = np.array([p.risk_probability for p in predictions], dtype=float)
    labels = np.array([p.label_geo_spread for p in predictions], dtype=float)

    total = len(probs)
    brier = np.mean((probs - labels) ** 2)

    bin_boundaries = np.linspace(0, 1, bins + 1)
    # np.digitize(x, bins) returns index i such that bins[i-1] <= x < bins[i]
    # For the last bin, we want it to be inclusive of 1.0
    bin_indices = np.digitize(probs, bin_boundaries[1:-1])

    ece = 0.0
    calibration_bins = []

    for i in range(bins):
        mask = bin_indices == i
        count = np.sum(mask)

        if count == 0:
            calibration_bins.append({
                "bin_start": float(bin_boundaries[i]),
                "bin_end": float(bin_boundaries[i+1]),
                "mean_prediction": 0.0,
                "observed_rate": 0.0,
                "count": 0.0,
            })
            continue

        mean_prediction = np.mean(probs[mask])
        observed_rate = np.mean(labels[mask])

        ece += (count / total) * abs(mean_prediction - observed_rate)

        calibration_bins.append({
            "bin_start": float(bin_boundaries[i]),
            "bin_end": float(bin_boundaries[i+1]),
            "mean_prediction": float(mean_prediction),
            "observed_rate": float(observed_rate),
            "count": float(count),
        })

    return {
        "expected_calibration_error": float(ece),
        "brier_score": float(brier),
        "n_bins": float(bins),
        "calibration_bins": calibration_bins,
    }
