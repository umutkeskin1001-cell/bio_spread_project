from typing import Any, Optional

import numpy as np
import polars as pl
from numpy.typing import NDArray


class BackboneGraphEmbedder:
    def __init__(self) -> None:
        self.encoder = None
        self.embeddings: Optional[NDArray[Any]] = None
        self.backbone_mapping: Optional[dict[str, int]] = None

    def fit(self, observations: pl.LazyFrame, split_year: int) -> None:
        import random
        import warnings

        import torch
        import torch.nn.functional as F

        warnings.filterwarnings(
            "ignore",
            message="`torch_geometric.distributed` has been deprecated*",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="`torch.jit.script` is deprecated*",
            category=DeprecationWarning,
        )
        from torch_geometric.data import Data

        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)

        # Vectorized data prep
        pre_obs = observations.filter(pl.col("year") <= split_year).collect()

        if "host_order" not in pre_obs.columns:
            pre_obs = pre_obs.with_columns(
                pl.col("host_genus").alias("host_order") if "host_genus" in pre_obs.columns else pl.lit("Unknown").alias("host_order")
            )

        # Node mappings
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

        # Edges
        # 1. Backbone-Country
        bc = pre_obs.with_columns([
            (pl.col("mobility_score") * (-0.2 * (split_year - pl.col("year"))).exp()).alias("w")
        ]).group_by(["backbone_id", "country"]).agg(pl.col("w").sum().alias("weight")).drop_nulls()

        bc_u = bc["backbone_id"].replace(b_map).to_numpy()
        bc_v = bc["country"].replace(c_map).to_numpy()
        bc_w = bc["weight"].to_numpy()

        # 2. Backbone-Host
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

            def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
                x = self.conv1(x, edge_index, edge_weight)
                x = F.relu(x)
                x = self.conv2(x, edge_index, edge_weight)
                return x

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

        b_ids = backbone_ids.to_list()
        d = self.embeddings.shape[1]

        # Vectorized mapping
        idx_series = np.array([self.backbone_mapping.get(str(bb), -1) for bb in b_ids], dtype=int)

        embeds = np.zeros((len(b_ids), d))
        mask = idx_series >= 0
        embeds[mask] = self.embeddings[idx_series[mask]]

        cols = {f"gnn_embed_{i}": embeds[:, i] for i in range(d)}
        cols["backbone_id"] = b_ids
        return pl.DataFrame(cols)
