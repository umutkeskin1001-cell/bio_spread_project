import numpy as np
import polars as pl
from scipy.sparse import coo_matrix, eye, diags

def build_fastrp_embeddings(records: pl.DataFrame, mash_df: pl.DataFrame, split_year: int) -> pl.DataFrame:
    """
    Constructs FastRP embeddings from a spatio-phylogenetic graph.
    Budget: ~30s
    """
    pre = records.filter(pl.col('year') <= split_year)
    backbone_ids = pre['backbone_id'].unique().sort().to_list()
    n = len(backbone_ids)
    if n == 0:
        return pl.DataFrame({'backbone_id': []}).with_columns(
            [pl.lit(0.0).alias(f'fastrp_{i}') for i in range(16)])
    
    id_to_idx = {bb: i for i, bb in enumerate(backbone_ids)}

    # 1. Genomic adjacency (Mash distance < 0.1)
    rows, cols, data = [], [], []
    mash = mash_df.filter(
        pl.col('backbone_id_1').is_in(backbone_ids) &
        pl.col('backbone_id_2').is_in(backbone_ids) &
        (pl.col('mash_distance') < 0.1)
    )
    for row in mash.iter_rows(named=True):
        a, b, d = row['backbone_id_1'], row['backbone_id_2'], row['mash_distance']
        i, j = id_to_idx[a], id_to_idx[b]
        w = np.exp(-d/0.01)
        rows.extend([i, j]); cols.extend([j, i]); data.extend([w, w])
    A = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

    # 2. Geographic adjacency (Shared country ratio)
    rows_g, cols_g, data_g = [], [], []
    bb_countries = pre.group_by("backbone_id").agg(pl.col("country").drop_nulls().unique())
    country_sets = {r["backbone_id"]: set(r["country"]) for r in bb_countries.to_dicts()}
    set_sizes = {bb: len(cset) for bb, cset in country_sets.items()}

    # Build candidate overlaps via country -> backbones index to avoid dense O(n^2) scan.
    country_to_backbones: dict[str, list[str]] = {}
    for bb, countries in country_sets.items():
        for country in countries:
            country_to_backbones.setdefault(str(country), []).append(str(bb))

    overlap_counts: dict[tuple[str, str], int] = {}
    for shared_backbones in country_to_backbones.values():
        uniq = sorted(set(shared_backbones))
        m = len(uniq)
        for a_idx in range(m):
            for b_idx in range(a_idx + 1, m):
                a = uniq[a_idx]
                b = uniq[b_idx]
                overlap_counts[(a, b)] = overlap_counts.get((a, b), 0) + 1

    for (bb_a, bb_b), inter in overlap_counts.items():
        maxs = max(set_sizes.get(bb_a, 0), set_sizes.get(bb_b, 0))
        if maxs <= 0:
            continue
        ratio = float(inter) / float(maxs)
        i = id_to_idx.get(bb_a)
        j = id_to_idx.get(bb_b)
        if i is None or j is None or i == j:
            continue
        w = ratio * 0.5
        rows_g.extend([i, j])
        cols_g.extend([j, i])
        data_g.extend([w, w])
    
    B = coo_matrix((data_g, (rows_g, cols_g)), shape=(n, n)).tocsr()
    C = 0.7 * A + 0.3 * B + eye(n)  # self-loops for connectivity

    # 3. FastRP Random Projection
    deg = np.array(C.sum(axis=1)).squeeze()
    D_inv_sqrt = diags(1.0 / np.sqrt(np.maximum(deg, 1e-8)))
    C_norm = D_inv_sqrt @ C @ D_inv_sqrt

    np.random.seed(42)
    emb_dim = 16
    R = np.random.randn(n, emb_dim) * 0.1
    embeddings = np.zeros((n, emb_dim))
    current = R.copy()
    for k, w in enumerate([1.0, 0.8, 0.6, 0.4]):
        embeddings += w * current
        if k < 3:
            current = C_norm @ current
            
    # Unit normalization
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    embeddings /= norm

    out = pl.DataFrame({'backbone_id': backbone_ids})
    for i in range(emb_dim):
        out = out.with_columns(pl.Series(f'fastrp_{i}', embeddings[:, i]))
    return out
