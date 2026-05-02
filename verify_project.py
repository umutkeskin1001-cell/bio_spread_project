#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
src_root = str(SRC_ROOT)
if src_root in sys.path:
    sys.path.remove(src_root)
sys.path.insert(0, src_root)

from verify_project import main

if __name__ == "__main__":
    raise SystemExit(main())
