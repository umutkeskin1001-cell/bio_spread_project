from __future__ import annotations

from pathlib import Path

import polars as pl


class EmbeddingStore:
    """Loads and caches pre-computed biological embeddings."""

    def __init__(self, dir_path: Path):
        self.dir_path = Path(dir_path)
        self.esm2 = pl.read_parquet(self.dir_path / "esm2_embeddings.parquet")
        nt_path = self.dir_path / "nt_embeddings.parquet"
        self.nt = pl.read_parquet(nt_path) if nt_path.exists() else None

    def get(self, backbone_ids: list[str], kind: str = "esm2") -> pl.DataFrame:
        if kind not in {"esm2", "nt"}:
            raise ValueError("kind must be one of: esm2, nt")
        table = self.esm2 if kind == "esm2" else self.nt
        if table is None:
            return pl.DataFrame({"backbone_id": backbone_ids})
        return table.filter(pl.col("backbone_id").is_in(backbone_ids))
