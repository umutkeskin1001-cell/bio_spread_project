from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import polars as pl
import torch
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, diags, eye
from sklearn.decomposition import NMF
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------
# synergy_features
# ---------------------------------------------------------------
INTERACTION_PAIRS = [
    ("T_eff_norm", "H_obs_specialization_norm"),
    ("T_eff_norm", "A_eff_norm"),
    ("H_obs_specialization_norm", "A_eff_norm"),
    ("H_external_host_range_norm", "replicon_architecture_norm"),
    ("orit_support", "amr_burden_saturation_norm"),
    ("backbone_purity_norm", "assignment_confidence_norm"),
    ("mean_antibiotic_pressure", "frac_pathogenic_hosts"),
]


def build_synergy_features(features: pl.DataFrame, pairs: list[tuple[str, str]]) -> pl.DataFrame:
    for a, b in pairs:
        if a in features.columns and b in features.columns:
            name = f"synergy_{a}__{b}"
            features = features.with_columns((pl.col(a) * pl.col(b)).alias(name))
    return features


# ---------------------------------------------------------------
# temporal_features
# ---------------------------------------------------------------
def _slope_from_series(years: list[int], values: list[float]) -> float:
    if len(years) < 2:
        return 0.0
    x = np.asarray(years, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    x_centered = x - x.mean()
    denom = float(np.sum(x_centered * x_centered))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(x_centered * (y - y.mean())) / denom)


def build_temporal_trend_features(records: pl.DataFrame, *, split_year: int) -> pl.DataFrame:
    if records.is_empty() or "backbone_id" not in records.columns:
        return pl.DataFrame({
            "backbone_id": [], "country_slope_train": [], "host_breadth_slope_train": [],
            "mobility_shift_slope_train": [], "recent_expansion_flag": [],
        })

    required = {"backbone_id", "year", "country", "host_genus", "mobility_score"}
    if not required.issubset(set(records.columns)):
        return pl.DataFrame({
            "backbone_id": records["backbone_id"].unique() if "backbone_id" in records.columns else [],
            "country_slope_train": [], "host_breadth_slope_train": [],
            "mobility_shift_slope_train": [], "recent_expansion_flag": [],
        })

    pre = records.filter(pl.col("year") <= split_year)
    if pre.is_empty():
        return pl.DataFrame({
            "backbone_id": records["backbone_id"].unique(),
            "country_slope_train": [0.0] * records["backbone_id"].n_unique(),
            "host_breadth_slope_train": [0.0] * records["backbone_id"].n_unique(),
            "mobility_shift_slope_train": [0.0] * records["backbone_id"].n_unique(),
            "recent_expansion_flag": [0] * records["backbone_id"].n_unique(),
        })

    out_rows: list[dict[str, float | int | str]] = []
    for bb_df in pre.sort(["backbone_id", "year"]).partition_by("backbone_id"):
        bb = str(bb_df["backbone_id"][0])
        year_min = int(bb_df["year"].min())
        years = list(range(year_min, split_year + 1))
        cum_countries: list[float] = []
        cum_hosts: list[float] = []
        mean_mob: list[float] = []
        seen_countries: set[str] = set()
        seen_hosts: set[str] = set()
        country_counts_by_year: dict[int, int] = {}

        for yr in years:
            year_slice = bb_df.filter(pl.col("year") == yr)
            countries = set(year_slice["country"].drop_nulls().cast(pl.Utf8).to_list())
            hosts = set(year_slice["host_genus"].drop_nulls().cast(pl.Utf8).to_list())
            seen_countries |= countries
            seen_hosts |= hosts
            cum_countries.append(float(len(seen_countries)))
            cum_hosts.append(float(len(seen_hosts)))
            if year_slice.height > 0:
                mean_mob.append(float(year_slice["mobility_score"].cast(pl.Float64).mean()))
            else:
                mean_mob.append(mean_mob[-1] if mean_mob else 0.0)
            country_counts_by_year[yr] = len(countries)

        recent_years = [split_year - 1, split_year]
        recent_expansion = int(any(country_counts_by_year.get(yr, 0) > 0 for yr in recent_years))

        out_rows.append({
            "backbone_id": bb, "country_slope_train": _slope_from_series(years, cum_countries),
            "host_breadth_slope_train": _slope_from_series(years, cum_hosts),
            "mobility_shift_slope_train": _slope_from_series(years, mean_mob),
            "recent_expansion_flag": recent_expansion,
        })

    return pl.DataFrame(out_rows)


# ---------------------------------------------------------------
# nmf_features
# ---------------------------------------------------------------
def build_nmf_diffusion_features(records: pl.LazyFrame, split_year: int) -> pl.DataFrame:
    pre_obs = records.filter(pl.col("year") <= split_year).collect()

    if pre_obs.is_empty():
        return pl.DataFrame({"backbone_id": [], "reach_potential": [], "saturation_deficit": []})

    pivot_df = pre_obs.group_by(["backbone_id", "country"]).len()
    pivot_wide = pivot_df.pivot(values="len", index="backbone_id", columns="country").fill_null(0)
    backbones = pivot_wide["backbone_id"].to_numpy()
    countries = [c for c in pivot_wide.columns if c != "backbone_id"]
    B = pivot_wide.select(countries).to_numpy()

    n_components = min(4, B.shape[0], B.shape[1])
    if n_components < 1:
        return pl.DataFrame({
            "backbone_id": backbones, "reach_potential": np.zeros(len(backbones)),
            "saturation_deficit": np.zeros(len(backbones)),
        })

    nmf = NMF(n_components=n_components, init='nndsvd' if min(B.shape) >= n_components else 'random', random_state=42)
    W = nmf.fit_transform(B)
    H = nmf.components_

    B_bin = (B > 0).astype(float)
    intersection = B_bin.T @ B_bin
    sz = B_bin.sum(axis=0)
    union = sz[:, None] + sz[None, :] - intersection
    union[union == 0] = 1.0
    C = intersection / union
    np.fill_diagonal(C, 1.0)

    alpha = 0.7
    H_smooth = alpha * H + (1 - alpha) * (C @ H.T).T
    Y_hat = W @ H_smooth

    threshold = 0.5
    current_counts = B_bin.sum(axis=1)
    predicted_counts = (Y_hat > threshold).sum(axis=1)

    reach_potential = np.clip(predicted_counts - current_counts, 0, None)
    safe_pred = np.maximum(predicted_counts, 1)
    saturation_deficit = np.clip(1.0 - (current_counts / safe_pred), 0.0, 1.0)

    return pl.DataFrame({
        "backbone_id": pl.Series(backbones),
        "reach_potential": pl.Series(reach_potential),
        "saturation_deficit": pl.Series(saturation_deficit),
    })


# ---------------------------------------------------------------
# grps
# ---------------------------------------------------------------
def compute_grps(
    low_knownness_ids: list[str], embeddings: pl.DataFrame, risk_labels: pl.DataFrame,
    embedding_cols: list[str], n_neighbors: int = 5,
) -> pl.DataFrame:
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
    nbrs = NearestNeighbors(metric="cosine", n_neighbors=min(k + 1, x_known.shape[0]))
    nbrs.fit(x_known)
    distances, indices = nbrs.kneighbors(x_query)

    eps = 1e-6
    scores: list[float] = []
    query_ids = query_embed["backbone_id"].to_list()
    known_ids = known_embed["backbone_id"].to_list()

    for idx_in_query, (d_row, i_row) in enumerate(zip(distances, indices)):
        q_id = query_ids[idx_in_query]
        mask = [known_ids[i] != q_id for i in i_row]
        d_filtered = d_row[mask][:k]
        i_filtered = i_row[mask][:k]

        if len(d_filtered) == 0:
            scores.append(0.0)
            continue

        w = 1.0 / (d_filtered + eps)
        risks = y_known[i_filtered]
        denom = float(np.sum(w))
        scores.append(float(np.sum(w * risks) / denom) if denom > 0 else 0.0)

    return pl.DataFrame({"backbone_id": query_embed["backbone_id"].to_list(), "grps": scores})


# ---------------------------------------------------------------
# bio_adapter
# ---------------------------------------------------------------
class BioAdapter(torch.nn.Module):  # type: ignore[misc]
    def __init__(self, input_dim: int = 1280, hidden: int = 64, latent_dim: int = 16) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, latent_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_bio_adapter(model_path: Path) -> BioAdapter:
    model = BioAdapter()
    if model_path.exists():
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
    model.eval()
    return model


def transform_bio_features(features_df: pl.DataFrame, esm_df: pl.DataFrame, adapter: BioAdapter) -> pl.DataFrame:
    latent_columns = [f"bio_adapt_{i}" for i in range(16)]
    if esm_df.is_empty():
        return features_df.with_columns([pl.lit(0.0).alias(c) for c in latent_columns])

    embed_cols = sorted([c for c in esm_df.columns if c.startswith("esm2_")])
    if not embed_cols:
        return features_df.with_columns([pl.lit(0.0).alias(c) for c in latent_columns])

    join_df = features_df.join(esm_df, on="backbone_id", how="inner")
    if join_df.is_empty():
        return features_df.with_columns([pl.lit(0.0).alias(c) for c in latent_columns])

    X: NDArray[np.float32] = np.asarray(join_df.select(embed_cols).to_numpy(), dtype=np.float32)
    with torch.no_grad():
        adapter.eval()
        latent_tensor = adapter(torch.from_numpy(X))
        latent: NDArray[np.float32] = np.asarray(latent_tensor.detach().cpu().numpy(), dtype=np.float32)

    latent_dim = int(latent.shape[1])
    latent_dim = min(latent_dim, len(latent_columns))
    latent_df = pl.DataFrame({
        "backbone_id": join_df["backbone_id"],
        **{latent_columns[i]: latent[:, i] for i in range(latent_dim)},
    })

    out = features_df.join(latent_df, on="backbone_id", how="left", coalesce=True)
    out = out.with_columns([pl.col(c).fill_null(0.0) for c in latent_columns])
    return out


# ---------------------------------------------------------------
# phylo_propagation
# ---------------------------------------------------------------
# ISSUE-09 FIX: Canonical implementation lives in phylo_propagation.py.
# This re-export ensures backward compatibility for any code importing from here.
from bio_spread_project.phylo_propagation import build_phylo_propagation as build_phylo_propagation  # noqa: F401


# ---------------------------------------------------------------
# gnn_embedder
# ---------------------------------------------------------------
class BackboneGraphEmbedder:
    def __init__(self) -> None:
        self.encoder = None
        self.embeddings: Optional[NDArray[Any]] = None
        self.backbone_mapping: Optional[dict[str, int]] = None

    def fit(self, observations: pl.LazyFrame, split_year: int) -> None:
        import torch.nn.functional as F
        from torch_geometric.data import Data

        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)

        pre_obs = observations.filter(pl.col("year") <= split_year).collect()

        if "host_order" not in pre_obs.columns:
            pre_obs = pre_obs.with_columns(
                pl.col("host_genus").alias("host_order") if "host_genus" in pre_obs.columns else pl.lit("Unknown").alias("host_order")
            )

        backbones = pre_obs["backbone_id"].drop_nulls().unique()
        countries = pre_obs["country"].drop_nulls().unique()
        hosts = pre_obs["host_order"].drop_nulls().unique()

        b_map = {id: i for i, id in enumerate(backbones)}
        c_offset = len(backbones)
        c_map = {id: i + c_offset for i, id in enumerate(countries)}
        h_offset = c_offset + len(countries)
        h_map = {id: i + h_offset for i, id in enumerate(hosts)}
        self.backbone_mapping = b_map
        num_nodes = h_offset + len(hosts)

        bc = pre_obs.with_columns([
            (pl.col("mobility_score") * (-0.2 * (split_year - pl.col("year"))).exp()).alias("w")
        ]).group_by(["backbone_id", "country"]).agg(pl.col("w").sum().alias("weight")).drop_nulls()

        bc_u = bc["backbone_id"].replace(b_map).to_numpy()
        bc_v = bc["country"].replace(c_map).to_numpy()
        bc_w = bc["weight"].to_numpy()

        bh = pre_obs.group_by(["backbone_id", "host_order"]).agg(pl.len().log1p().alias("weight")).drop_nulls()
        bh_u = bh["backbone_id"].replace(b_map).to_numpy()
        bh_v = bh["host_order"].replace(h_map).to_numpy()
        bh_w = bh["weight"].to_numpy()

        src = np.concatenate([bc_u, bc_v, bh_u, bh_v])
        dst = np.concatenate([bc_v, bc_u, bh_v, bh_u])
        weights = np.concatenate([bc_w, bc_w, bh_w, bh_w])

        edge_index = torch.from_numpy(np.vstack([src, dst]).astype(np.int64))
        edge_weight = torch.tensor(weights, dtype=torch.float)
        x = torch.randn((num_nodes, 16))

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
        from torch_geometric.nn import GCNConv

        class GNN(torch.nn.Module):  # type: ignore[misc]
            def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
                super().__init__()
                self.conv1 = GCNConv(in_channels, hidden_channels)
                self.conv2 = GCNConv(hidden_channels, out_channels)

            def forward(self, x_t: torch.Tensor, ei: torch.Tensor, ew: torch.Tensor) -> torch.Tensor:
                h = self.conv1(x_t, ei, ew)
                h = F.relu(h)
                return self.conv2(h, ei, ew)

        model = GNN(16, 16, 8)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        model.train()
        for _ in range(20):
            optimizer.zero_grad()
            z = model(data.x, data.edge_index, data.edge_attr)
            if edge_index.size(1) > 0:
                idx_sample = torch.randint(0, edge_index.size(1), (min(5000, edge_index.size(1)),))
                u_batch, v_batch = edge_index[0, idx_sample], edge_index[1, idx_sample]
                pos_score = (z[u_batch] * z[v_batch]).sum(dim=-1)
                neg_v = torch.randint(0, num_nodes, (u_batch.size(0),))
                neg_score = (z[u_batch] * z[neg_v]).sum(dim=-1)
                loss = torch.relu(1.0 - pos_score + neg_score).mean()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            self.embeddings = model(data.x, data.edge_index, data.edge_attr).numpy()

    def transform(self, backbone_ids: pl.Series) -> pl.DataFrame:
        if self.embeddings is None:
            raise ValueError("Model not fitted")
        if self.backbone_mapping is None:
            raise ValueError("Backbone mapping missing; call fit() before transform()")

        b_ids = backbone_ids.to_list()
        d = self.embeddings.shape[1]

        idx_series = backbone_ids.replace(self.backbone_mapping).fill_null(-1).to_numpy().astype(int)
        embeds = np.zeros((len(b_ids), d))
        mask = idx_series >= 0
        embeds[mask] = self.embeddings[idx_series[mask]]

        cols = {f"gnn_embed_{i}": embeds[:, i] for i in range(d)}
        cols["backbone_id"] = b_ids
        return pl.DataFrame(cols)


# ---------------------------------------------------------------
# phylo_spatial_embedder
# ---------------------------------------------------------------
class GCN(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int, d_hidden: int = 16) -> None:
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(d_in, d_hidden)
        self.conv2 = GCNConv(d_hidden, d_out)

    def forward(self, x_t: torch.Tensor, e_idx: torch.Tensor, e_w: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        h = self.conv1(x_t, e_idx, e_w)
        h = F.relu(h)
        return self.conv2(h, e_idx, e_w)

class PhyloSpatialGraphEmbedder:
    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.backbone_map: dict[str, int] = {}
        self.embeddings: NDArray[np.float32] | None = None
        self.model: torch.nn.Module | None = None

    def fit(
        self, records: pl.DataFrame, *, split_year: int, backbone_ids: list[str],
        feature_matrix: NDArray[np.floating] | NDArray[np.integer],
        mash_path: Path | None = None
    ) -> PhyloSpatialGraphEmbedder:
        import torch.nn.functional as F
        from torch_geometric.data import Data
        
        self.backbone_map = {bb: i for i, bb in enumerate(backbone_ids)}
        n_backbones = len(backbone_ids)
        if n_backbones == 0:
            return self

        pre = records.filter(pl.col("year") <= split_year)
        edges: list[tuple[int, int, float]] = []

        if not pre.is_empty() and {"backbone_id", "country"}.issubset(pre.columns):
            bc = pre.select(["backbone_id", "country"]).drop_nulls().unique().group_by("country").agg(pl.col("backbone_id"))
            for row in bc.to_dicts():
                bbs = [b for b in row["backbone_id"] if b in self.backbone_map]
                for i in range(len(bbs)):
                    for j in range(i + 1, len(bbs)):
                        edges.append((self.backbone_map[bbs[i]], self.backbone_map[bbs[j]], 1.0))

        if not pre.is_empty() and {"backbone_id", "host_genus"}.issubset(pre.columns):
            bh = pre.select(["backbone_id", "host_genus"]).drop_nulls().unique().group_by("host_genus").agg(pl.col("backbone_id"))
            for row in bh.to_dicts():
                bbs = [b for b in row["backbone_id"] if b in self.backbone_map]
                for i in range(len(bbs)):
                    for j in range(i + 1, len(bbs)):
                        edges.append((self.backbone_map[bbs[i]], self.backbone_map[bbs[j]], 0.8))

        if mash_path is not None and mash_path.exists():
            mash = pl.read_csv(mash_path, separator="\t") if mash_path.suffix.lower() in {".tsv", ".tab"} else pl.read_csv(mash_path)
            req = {"backbone_id_1", "backbone_id_2", "mash_distance"}
            if req.issubset(set(mash.columns)):
                for row in mash.filter(pl.col("mash_distance") < 0.2).to_dicts():
                    a = row["backbone_id_1"]
                    b = row["backbone_id_2"]
                    if a in self.backbone_map and b in self.backbone_map:
                        w = float(np.exp(-float(row["mash_distance"])))
                        edges.append((self.backbone_map[a], self.backbone_map[b], w))

        if not edges:
            self.embeddings = np.zeros((n_backbones, self.dim), dtype=np.float32)
            return self

        src, dst, wts = [], [], []
        for u, v, w in edges:
            src.extend([u, v])
            dst.extend([v, u])
            wts.extend([w, w])

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(wts, dtype=torch.float32)
        fm = np.asarray(feature_matrix, dtype=np.float32)
        in_dim = min(fm.shape[1], 32)
        x = torch.tensor(fm[:, :in_dim], dtype=torch.float32)
        if x.shape[1] < 32:
            pad = torch.zeros((x.shape[0], 32 - x.shape[1]), dtype=torch.float32)
            x = torch.cat([x, pad], dim=1)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        self.model = GCN(32, self.dim)
        opt = torch.optim.Adam(self.model.parameters(), lr=0.01)

        self.model.train()
        for _ in range(30):
            opt.zero_grad()
            z_t = self.model(data.x, data.edge_index, data.edge_attr)
            idx = torch.randint(0, data.edge_index.shape[1], (min(5000, data.edge_index.shape[1]),))
            u = data.edge_index[0, idx]
            v = data.edge_index[1, idx]
            pos = (z_t[u] * z_t[v]).sum(dim=-1)
            neg_v = torch.randint(0, n_backbones, (u.shape[0],))
            neg = (z_t[u] * z_t[neg_v]).sum(dim=-1)
            loss = torch.relu(1.0 - pos + neg).mean()
            loss.backward()
            opt.step()
        
        self.model.eval()
        with torch.no_grad():
            self.embeddings = np.asarray(self.model(data.x, data.edge_index, data.edge_attr).cpu().numpy(), dtype=np.float32)
        return self

    def transform(self, backbone_ids: list[str], feature_matrix: NDArray[np.floating] | NDArray[np.integer]) -> pl.DataFrame:
        if self.model is None:
            # Fallback if fit failed or had no edges
            n = len(backbone_ids)
            cols = {f"psge_{i}": np.zeros(n) for i in range(self.dim)}
            cols["backbone_id"] = backbone_ids
            return pl.DataFrame(cols)
        
        # In inductive mode, if we don't have edges for the transform set, we use zero-connectivity
        # or we could use the feature matrix and pass it through the GCN layers without neighbors (self-loops only)
        import torch
        self.model.eval()
        fm = np.asarray(feature_matrix, dtype=np.float32)
        x = torch.tensor(fm[:, : min(fm.shape[1], 32)], dtype=torch.float32)
        if x.shape[1] < 32:
            pad = torch.zeros((x.shape[0], 32 - x.shape[1]), dtype=torch.float32)
            x = torch.cat([x, pad], dim=1)
        
        # Dummy empty edge index for inductive transform on isolated nodes
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0,), dtype=torch.float32)
        
        with torch.no_grad():
            z = self.model(x, edge_index, edge_attr).cpu().numpy()
        
        cols = {f"psge_{i}": z[:, i] for i in range(self.dim)}
        cols["backbone_id"] = backbone_ids
        return pl.DataFrame(cols)

    def fit_transform(
        self, records: pl.DataFrame, *, split_year: int, backbone_ids: list[str],
        feature_matrix: NDArray[np.floating] | NDArray[np.integer],
        mash_path: Path | None = None, save_path: Path | None = None,
    ) -> pl.DataFrame:
        self.fit(records, split_year=split_year, backbone_ids=backbone_ids, feature_matrix=feature_matrix, mash_path=mash_path)
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"backbone_map": self.backbone_map, "embeddings": self.embeddings, "state_dict": self.model.state_dict() if self.model else None}, save_path)
        
        return self._to_df(backbone_ids)

    def _to_df(self, backbone_ids: list[str]) -> pl.DataFrame:
        if self.embeddings is None:
            raise ValueError("embeddings are not fitted")
        
        # Create a mapping for efficient lookup
        id_to_emb = {}
        if hasattr(self, 'backbone_map'):
            for b_id, idx in self.backbone_map.items():
                id_to_emb[b_id] = self.embeddings[idx]
        
        dim = self.embeddings.shape[1]
        results = []
        for b_id in backbone_ids:
            if b_id in id_to_emb:
                results.append(id_to_emb[b_id])
            else:
                results.append(np.zeros(dim))
        
        results_arr = np.array(results)
        cols: dict[str, Any] = {f"psge_{i}": results_arr[:, i] for i in range(dim)}
        cols["backbone_id"] = backbone_ids
        return pl.DataFrame(cols)


# ---------------------------------------------------------------
# graph_contagion
# ---------------------------------------------------------------
def build_fastrp_embeddings(records: pl.DataFrame, mash_df: pl.DataFrame, split_year: int) -> pl.DataFrame:
    pre = records.filter(pl.col('year') <= split_year)
    backbone_ids = pre['backbone_id'].unique().sort().to_list()
    n = len(backbone_ids)
    if n == 0:
        return pl.DataFrame({'backbone_id': []}).with_columns([pl.lit(0.0).alias(f'fastrp_{i}') for i in range(16)])

    id_to_idx = {bb: i for i, bb in enumerate(backbone_ids)}

    rows, cols, data = [], [], []
    mash = mash_df.filter(pl.col('backbone_id_1').is_in(backbone_ids) & pl.col('backbone_id_2').is_in(backbone_ids) & (pl.col('mash_distance') < 0.1))
    for row in mash.iter_rows(named=True):
        a, b, d = row['backbone_id_1'], row['backbone_id_2'], row['mash_distance']
        i, j = id_to_idx[a], id_to_idx[b]
        w = np.exp(-d / 0.01)
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([w, w])
    A = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

    bb_country_pairs = pre.select(["backbone_id", "country"]).unique()
    all_countries = sorted(bb_country_pairs["country"].unique().to_list())
    country_to_idx = {c: i for i, c in enumerate(all_countries)}
    bb_to_idx = {bb: i for i, bb in enumerate(backbone_ids)}
    country_matrix = np.zeros((n, len(all_countries)), dtype=np.float64)
    for row in bb_country_pairs.to_dicts():
        bi = bb_to_idx.get(row["backbone_id"])
        ci = country_to_idx.get(row["country"])
        if bi is not None and ci is not None:
            country_matrix[bi, ci] = 1.0
    geo_adj_dense = country_matrix @ country_matrix.T
    row_sums = country_matrix.sum(axis=1, keepdims=True)
    max_sums = np.maximum(row_sums, row_sums.T)
    with np.errstate(divide='ignore', invalid='ignore'):
        geo_adj_norm = np.where(max_sums > 0, geo_adj_dense / max_sums, 0.0) * 0.5
    rows_g, cols_g = np.where(geo_adj_norm > 0)
    data_g = geo_adj_norm[rows_g, cols_g]
    B = coo_matrix((data_g, (rows_g, cols_g)), shape=(n, n)).tocsr()
    C = 0.7 * A + 0.3 * B + eye(n)

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

    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    embeddings /= norm

    out = pl.DataFrame({'backbone_id': backbone_ids})
    for i in range(emb_dim):
        out = out.with_columns(pl.Series(f'fastrp_{i}', embeddings[:, i]))
    return out
