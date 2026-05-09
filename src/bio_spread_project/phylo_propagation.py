"""
Phylo-Propagation Module — Canonical Implementation
====================================================

Label Propagation over Mash-distance phylogenetic graphs.
This is the SINGLE source of truth for phylo-propagation logic.

Mathematical Design:
  Given a symmetric Mash-distance graph G = (V, E, w) where
  w(u,v) = exp(-mash_distance(u,v)), we compute the normalized
  Laplacian-smoothed label propagation:

    F^(t+1) = (1-α)·Y + α·S·F^(t)

  where S = D^{-½}·A·D^{-½} is the symmetric normalized adjacency,
  and Y is the initial label matrix.

Zero-Leakage Contract:
  - When labeled_ids is provided, ONLY those IDs contribute to Y.
  - When labeled_ids is provided, graph edges are FILTERED to only
    include connections between labeled nodes. Unlabeled nodes receive
    risk via inductive nearest-labeled-node smoothing AFTER propagation,
    preventing transductive graph shortcuts.
"""

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, diags


class PhyloPropagator:
    """Stateful phylo-propagation model for production inference.

    Stores the training label map and mash distances so that at inference
    time, risk can be propagated from known (training) labels to unseen
    backbone IDs without needing the original training DataFrame.
    """

    def __init__(
        self,
        split_year: int,
        alpha: float = 0.99,
        max_iter: int = 20,
        distance_threshold: float = 0.2,
    ):
        self.split_year = split_year
        self.alpha = alpha
        self.max_iter = max_iter
        self.distance_threshold = distance_threshold
        self.mash_path: Optional[Path] = None
        self.train_df: Optional[pl.DataFrame] = None

    def predict(
        self,
        graph: Any,
        features: pl.DataFrame,
        mask: Optional[NDArray[np.bool_]] = None,
    ) -> pl.DataFrame:
        """
        Calculates risk using label propagation.

        Args:
            graph: Unused legacy parameter (kept for API compatibility).
            features: DataFrame with backbone_id (and label_geo_spread if mask is used).
            mask: Boolean array indicating which rows have usable labels.
                  If provided, only those labels are used in propagation.
        """
        if self.mash_path is None:
            return pl.DataFrame(
                {"backbone_id": features["backbone_id"], "phylo_prop_risk": 0.5}
            )

        if mask is not None:
            # Cross-validation mode: use only the labels allowed by the mask
            labeled_ids = set(
                features.filter(pl.lit(mask))["backbone_id"].to_list()
            )
            return build_phylo_propagation(
                features,
                self.mash_path,
                self.split_year,
                alpha=self.alpha,
                max_iter=self.max_iter,
                distance_threshold=self.distance_threshold,
                labeled_ids=labeled_ids,
            )
        else:
            # Inference mode: join holdout features with stored training labels
            if self.train_df is None:
                return pl.DataFrame(
                    {"backbone_id": features["backbone_id"], "phylo_prop_risk": 0.5}
                )

            # Combine training data (labels) with holdout data (for graph construction)
            combined = pl.concat(
                [
                    self.train_df.select(["backbone_id", "label_geo_spread"]),
                    features.select(["backbone_id"]).with_columns(
                        pl.lit(None).alias("label_geo_spread")
                    ),
                ]
            ).unique("backbone_id")

            labeled_ids = set(self.train_df["backbone_id"].to_list())

            # Run propagation on the combined set
            prop_df = build_phylo_propagation(
                combined,
                self.mash_path,
                self.split_year,
                alpha=self.alpha,
                max_iter=self.max_iter,
                distance_threshold=self.distance_threshold,
                labeled_ids=labeled_ids,
            )

            # Return only for the requested features
            return features.select("backbone_id").join(
                prop_df, on="backbone_id", how="left"
            )


def build_phylo_propagation(
    features: pl.DataFrame,
    mash_path: Union[str, Path],
    split_year: int,
    alpha: float = 0.99,
    max_iter: int = 20,
    distance_threshold: float = 0.2,
    labeled_ids: Optional[set[str]] = None,
) -> pl.DataFrame:
    """
    Returns a DataFrame with columns: backbone_id, phylo_prop_risk.

    Zero-Leakage Contract:
    ──────────────────────
    When labeled_ids is provided:
      1. Only IDs in labeled_ids contribute labels to Y.
      2. The propagation graph S is built using ONLY edges between labeled nodes.
         This prevents transductive shortcuts where train labels propagate through
         val-node connectivity and bounce back with inflated signal.
      3. Unlabeled nodes receive risk via inductive nearest-labeled-node smoothing
         (weighted average of their k-nearest labeled neighbors' propagated risk).

    When labeled_ids is None:
      - Uses StratifiedKFold to generate honest OOF risk estimates.
    """
    try:
        mash_df = pl.read_csv(
            mash_path,
            separator="\t",
            has_header=True,
            columns=["backbone_id_1", "backbone_id_2", "mash_distance"],
        )
    except Exception:
        try:
            mash_df = pl.read_csv(mash_path, separator="\t", has_header=True)
            rename_map = {}
            if "backbone_id_a" in mash_df.columns:
                rename_map["backbone_id_a"] = "backbone_id_1"
            if "backbone_id_b" in mash_df.columns:
                rename_map["backbone_id_b"] = "backbone_id_2"
            if "distance" in mash_df.columns:
                rename_map["distance"] = "mash_distance"
            if rename_map:
                mash_df = mash_df.rename(rename_map)
        except Exception as exc:
            raise ValueError(
                f"Failed to parse mash-distance graph from `{mash_path}`. "
                "Refusing silent neutral-risk fallback."
            ) from exc

    mash_df = mash_df.filter(pl.col("mash_distance") < distance_threshold)

    all_ids = sorted(features["backbone_id"].unique().to_list())
    id_to_idx = {bb: i for i, bb in enumerate(all_ids)}
    n = len(all_ids)

    if n == 0:
        return pl.DataFrame({"backbone_id": [], "phylo_prop_risk": []})

    # ──────────────────────────────────────────────────────────────────
    # Build the label vector Y
    # ──────────────────────────────────────────────────────────────────
    Y = np.zeros((n, 2))
    label_map = dict(zip(features["backbone_id"], features["label_geo_spread"]))
    labeled_idx_set = set()  # Track which indices are labeled

    for i, bb in enumerate(all_ids):
        # Leakage Control: Only use label if ID is in the allowed set
        if labeled_ids is not None and bb not in labeled_ids:
            continue

        lab = label_map.get(bb)
        if lab == 1:
            Y[i, 1] = 1.0
            labeled_idx_set.add(i)
        elif lab == 0:
            Y[i, 0] = 1.0
            labeled_idx_set.add(i)

    if labeled_ids is not None:
        # ──────────────────────────────────────────────────────────────
        # INDUCTIVE MODE (fold-isolated): Build graph ONLY between
        # labeled nodes to prevent transductive shortcuts.
        # ──────────────────────────────────────────────────────────────
        rows, cols, data = [], [], []
        # Also store raw distances for inductive projection to unlabeled nodes
        unlabeled_distances: dict[int, list[tuple[int, float]]] = {}

        for row in mash_df.iter_rows(named=True):
            a, b, d = row["backbone_id_1"], row["backbone_id_2"], row["mash_distance"]
            if a in id_to_idx and b in id_to_idx:
                i, j = id_to_idx[a], id_to_idx[b]
                w = np.exp(-d)

                i_labeled = i in labeled_idx_set
                j_labeled = j in labeled_idx_set

                if i_labeled and j_labeled:
                    # Both labeled: include in propagation graph
                    rows.extend([i, j])
                    cols.extend([j, i])
                    data.extend([w, w])
                elif i_labeled and not j_labeled:
                    # One labeled, one unlabeled: store for inductive projection
                    unlabeled_distances.setdefault(j, []).append((i, d))
                elif j_labeled and not i_labeled:
                    unlabeled_distances.setdefault(i, []).append((j, d))

        if not rows:
            # No labeled edges — fall back to prior (0.5)
            return pl.DataFrame(
                {"backbone_id": all_ids, "phylo_prop_risk": [0.5] * n}
            )

        A = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        deg = np.array(A.sum(axis=1)).flatten()
        deg_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-8))
        D_half_inv = diags(deg_inv_sqrt)
        S = D_half_inv @ A @ D_half_inv

        # Propagate on the labeled-only subgraph
        F = Y.copy()
        for _ in range(max_iter):
            F = (1 - alpha) * Y + alpha * (S @ F)

        risk = F[:, 1]

        # Inductive projection for unlabeled nodes:
        # Weighted average of their nearest labeled neighbors' propagated risk
        for u_idx, neighbors in unlabeled_distances.items():
            if not neighbors:
                continue
            weights = np.array([np.exp(-d) for _, d in neighbors])
            neighbor_risks = np.array([risk[l_idx] for l_idx, _ in neighbors])
            w_sum = weights.sum()
            if w_sum > 0:
                risk[u_idx] = float(np.dot(weights, neighbor_risks) / w_sum)
            else:
                risk[u_idx] = 0.5

    else:
        # ──────────────────────────────────────────────────────────────
        # OOF MODE: Full graph, but labels are masked per fold.
        # ──────────────────────────────────────────────────────────────
        rows_all, cols_all, data_all = [], [], []
        for row in mash_df.iter_rows(named=True):
            a, b, d = row["backbone_id_1"], row["backbone_id_2"], row["mash_distance"]
            if a in id_to_idx and b in id_to_idx:
                i, j = id_to_idx[a], id_to_idx[b]
                w = np.exp(-d)
                rows_all.extend([i, j])
                cols_all.extend([j, i])
                data_all.extend([w, w])

        if not rows_all:
            return pl.DataFrame(
                {"backbone_id": all_ids, "phylo_prop_risk": [0.5] * n}
            )

        A = coo_matrix((data_all, (rows_all, cols_all)), shape=(n, n)).tocsr()
        deg = np.array(A.sum(axis=1)).flatten()
        deg_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-8))
        D_half_inv = diags(deg_inv_sqrt)
        S = D_half_inv @ A @ D_half_inv

        from sklearn.model_selection import StratifiedKFold

        risk = np.zeros(n)
        y_labels = Y[:, 1].astype(int)
        class_counts = np.bincount(y_labels, minlength=2)
        min_class = int(class_counts.min())
        if min_class < 2 or len(np.unique(y_labels)) < 2:
            return pl.DataFrame(
                {"backbone_id": all_ids, "phylo_prop_risk": [0.5] * n}
            )
        n_splits = min(5, n, min_class)

        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(Y, y_labels):
            Y_fold = Y.copy()
            Y_fold[val_idx, :] = 0.0  # Strictly mask out evaluation labels
            F_fold = Y_fold.copy()
            for _ in range(max_iter):
                F_fold = (1 - alpha) * Y_fold + alpha * (S @ F_fold)
            risk[val_idx] = F_fold[val_idx, 1]

    # Robust normalization to [0, 1] via sigmoid centering
    rmin, rmax = risk.min(), risk.max()
    if rmax > rmin:
        risk = (risk - rmin) / (rmax - rmin)

    return pl.DataFrame({"backbone_id": all_ids, "phylo_prop_risk": risk})
