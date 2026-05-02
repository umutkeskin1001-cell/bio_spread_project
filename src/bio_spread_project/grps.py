from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.neighbors import NearestNeighbors


def compute_grps(
    low_knownness_ids: list[str],
    embeddings: pl.DataFrame,
    risk_labels: pl.DataFrame,
    embedding_cols: list[str],
    n_neighbors: int = 5,
) -> pl.DataFrame:
    """Returns DataFrame with columns: backbone_id, grps."""
    if not low_knownness_ids:
        return pl.DataFrame({"backbone_id": [], "grps": []}, schema={"backbone_id": pl.Utf8, "grps": pl.Float64})

    known_labels = risk_labels.filter(pl.col("label_geo_spread").is_not_null()).select(["backbone_id", "label_geo_spread"])
    known_embed = embeddings.join(known_labels, on="backbone_id", how="inner")
    if known_embed.is_empty():
        return pl.DataFrame({"backbone_id": low_knownness_ids, "grps": [0.0] * len(low_knownness_ids)})

    query_embed = embeddings.filter(pl.col("backbone_id").is_in(low_knownness_ids))
    if query_embed.is_empty():
        return pl.DataFrame({"backbone_id": low_knownness_ids, "grps": [0.0] * len(low_knownness_ids)})

    x_known = known_embed.select(embedding_cols).fill_null(0.0).to_numpy()
    y_known = known_embed["label_geo_spread"].cast(pl.Float64).to_numpy()
    x_query = query_embed.select(embedding_cols).fill_null(0.0).to_numpy()

    k = max(1, min(n_neighbors, x_known.shape[0]))
    nbrs = NearestNeighbors(metric="cosine", n_neighbors=k)
    nbrs.fit(x_known)
    distances, indices = nbrs.kneighbors(x_query)

    eps = 1e-6
    scores: list[float] = []
    for d_row, i_row in zip(distances, indices):
        w = 1.0 / (d_row + eps)
        risks = y_known[i_row]
        denom = float(np.sum(w))
        scores.append(float(np.sum(w * risks) / denom) if denom > 0 else 0.0)

    return pl.DataFrame({"backbone_id": query_embed["backbone_id"].to_list(), "grps": scores})
