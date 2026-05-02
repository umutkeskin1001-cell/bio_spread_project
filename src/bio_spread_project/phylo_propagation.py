import numpy as np
import polars as pl
from scipy.sparse import coo_matrix, diags
from pathlib import Path

def build_phylo_propagation(
    features: pl.DataFrame,
    mash_path: str | Path,
    split_year: int,
    alpha: float = 0.99,
    max_iter: int = 20,
    distance_threshold: float = 0.2,
    labeled_ids: set[str] | None = None,
) -> pl.DataFrame:
    """
    Returns a DataFrame with columns: backbone_id, phylo_prop_risk.
    If labeled_ids is provided, only use labels from those IDs in the Y matrix.
    This prevents target leakage by masking the labels of query backbones.
    """
    try:
        mash_df = pl.read_csv(mash_path, separator="\t", has_header=True,
                              columns=["backbone_id_1", "backbone_id_2", "mash_distance"])
    except Exception:
        try:
            mash_df = pl.read_csv(mash_path, separator="\t", has_header=True)
            rename_map = {}
            if "backbone_id_a" in mash_df.columns: rename_map["backbone_id_a"] = "backbone_id_1"
            if "backbone_id_b" in mash_df.columns: rename_map["backbone_id_b"] = "backbone_id_2"
            if "distance" in mash_df.columns: rename_map["distance"] = "mash_distance"
            if rename_map:
                mash_df = mash_df.rename(rename_map)
        except Exception:
            return pl.DataFrame({"backbone_id": features["backbone_id"].unique(), "phylo_prop_risk": 0.5})
    
    mash_df = mash_df.filter(pl.col("mash_distance") < distance_threshold)
    
    all_ids = sorted(features["backbone_id"].unique().to_list())
    id_to_idx = {bb: i for i, bb in enumerate(all_ids)}
    n = len(all_ids)
    
    if n == 0:
        return pl.DataFrame({"backbone_id": [], "phylo_prop_risk": []})

    rows, cols, data = [], [], []
    for row in mash_df.iter_rows(named=True):
        a, b, d = row["backbone_id_1"], row["backbone_id_2"], row["mash_distance"]
        if a in id_to_idx and b in id_to_idx:
            i, j = id_to_idx[a], id_to_idx[b]
            w = np.exp(-d)
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([w, w])
    
    if not rows:
        return pl.DataFrame({"backbone_id": all_ids, "phylo_prop_risk": [0.5] * n})

    A = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-8))
    D_half_inv = diags(deg_inv_sqrt)
    S = D_half_inv @ A @ D_half_inv
    
    Y = np.zeros((n, 2))
    label_map = dict(zip(features["backbone_id"], features["label_geo_spread"]))
    for i, bb in enumerate(all_ids):
        # Leakage Control: Only use label if ID is explicitly provided in the allowed 'labeled_ids' set
        if labeled_ids is not None and bb not in labeled_ids:
            continue
            
        lab = label_map.get(bb)
        if lab == 1:
            Y[i, 1] = 1.0
        elif lab == 0:
            Y[i, 0] = 1.0
    
    F = Y.copy()
    if labeled_ids is not None:
        for _ in range(max_iter):
            F = (1 - alpha) * Y + alpha * (S @ F)
        risk = F[:, 1]
    else:
        from sklearn.model_selection import KFold
        risk = np.zeros(n)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(Y):
            Y_fold = Y.copy()
            Y_fold[val_idx, :] = 0.0  # Mask out validation labels
            F_fold = Y_fold.copy()
            for _ in range(max_iter):
                F_fold = (1 - alpha) * Y_fold + alpha * (S @ F_fold)
            risk[val_idx] = F_fold[val_idx, 1]
            
    # Simple normalization to [0, 1]
    rmin, rmax = risk.min(), risk.max()
    if rmax > rmin:
        risk = (risk - rmin) / (rmax - rmin)
    
    return pl.DataFrame({"backbone_id": all_ids, "phylo_prop_risk": risk})
