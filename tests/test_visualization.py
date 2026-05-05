from __future__ import annotations

import matplotlib


def test_visualization_uses_non_interactive_backend() -> None:
    from bio_spread_project import visualization  # noqa: F401

    assert matplotlib.get_backend().lower() == "agg"
