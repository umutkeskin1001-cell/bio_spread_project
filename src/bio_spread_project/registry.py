"""Model registry helpers for run-level tracking."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bio_spread_project.io_utils import ensure_directory


def append_model_registry_entry(path: str | Path, entry: dict[str, Any]) -> Path:
    registry_path = Path(path)
    ensure_directory(registry_path.parent)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **entry,
    }
    with registry_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return registry_path
