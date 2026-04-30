from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Sequence

import polars as pl


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_dataclass_csv(path: Path, data: pl.DataFrame | Sequence[Any]) -> Path:
    ensure_directory(path.parent)
    df = data if isinstance(data, pl.DataFrame) else pl.DataFrame(data)
    for name, dtype in zip(df.columns, df.dtypes):
        if str(dtype).startswith("Struct") or str(dtype).startswith("List"):
            df = df.with_columns(pl.col(name).map_elements(lambda value: json.dumps(value), return_dtype=pl.String).alias(name))

    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as tmp:
        df.write_csv(tmp.name)
        temp_path = Path(tmp.name)
    _replace_atomic(temp_path, path)
    return path


def write_dataclass_parquet(path: Path, data: pl.DataFrame | Sequence[Any]) -> Path:
    ensure_directory(path.parent)
    df = data if isinstance(data, pl.DataFrame) else pl.DataFrame(data)

    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as tmp:
        df.write_parquet(tmp.name, compression="zstd")
        temp_path = Path(tmp.name)
    _replace_atomic(temp_path, path)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def write_text(path: Path, content: str) -> Path:
    ensure_directory(path.parent)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(content)
        temp_path = Path(handle.name)
    _replace_atomic(temp_path, path)
    return path


def sha256_file(path: str | Path) -> str:
    """Return a SHA-256 hash for a local file using streaming chunks."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_atomic(temp_path: Path, target_path: Path) -> None:
    temp_path.replace(target_path)
