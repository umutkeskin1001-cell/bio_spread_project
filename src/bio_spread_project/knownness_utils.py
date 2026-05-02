from __future__ import annotations


def knownness_threshold(score: float, low: float = 0.25, high: float = 0.55) -> str:
    """Maps knownness score to 'low', 'medium', 'high'."""
    if score < low:
        return "low"
    if score < high:
        return "medium"
    return "high"
