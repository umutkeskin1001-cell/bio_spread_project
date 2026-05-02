import polars as pl
import numpy as np

class BackboneGraphEmbedder:
    def __init__(self):
        self.encoder = None
        self.embeddings = None
        self.backbone_mapping = None

    def fit(self, observations: pl.LazyFrame, split_year: int) -> None:
        import torch
        import random
        from torch_geometric.data import Data
        from torch_geometric.nn import SAGEConv
        import torch.nn.functional as F

        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)

        pre_obs = observations.filter(pl.col("year") <= split_year).collect()
        
        # We need host_order. If it's missing, let's just use host_genus or "Unknown"
        if "host_order" not in pre_obs.columns:
            if "host_genus" in pre_obs.columns:
                pre_obs = pre_obs.with_columns(pl.col("host_genus").alias("host_order"))
            else:
                pre_obs = pre_obs.with_columns(pl.lit("Unknown").alias("host_order"))

        # Backbone-Country
        bc = pre_obs.with_columns([
            (pl.col("mobility_score") * (-0.2 * (split_year - pl.col("year"))).exp()).alias("w")
        ]).group_by(["backbone_id", "country"]).agg(pl.col("w").sum().alias("weight"))

        # Backbone-HostOrder
        bh = pre_obs.group_by(["backbone_id", "host_order"]).agg(
            pl.len().log1p().alias("weight")
        )

        # Country-Country (co-occurrence proxy)
        countries = pre_obs["country"].drop_nulls().unique().to_list()
        n_c = len(countries)
        cc_weight = 1.0 / max(1, n_c)
        cc_src, cc_dst = [], []
        for i, c1 in enumerate(countries):
            for j, c2 in enumerate(countries):
                if i != j:
                    cc_src.append(c1)
                    cc_dst.append(c2)
        cc = pl.DataFrame({"c1": cc_src, "c2": cc_dst}).with_columns(pl.lit(cc_weight).alias("weight"))

        # Node mappings
        backbones = pre_obs["backbone_id"].drop_nulls().unique().to_list()
        hosts = pre_obs["host_order"].drop_nulls().unique().to_list()
        
        node_to_idx = {}
        idx = 0
        for b in backbones:
            node_to_idx[f"b_{b}"] = idx
            idx += 1
        for c in countries:
            node_to_idx[f"c_{c}"] = idx
            idx += 1
        for h in hosts:
            node_to_idx[f"h_{h}"] = idx
            idx += 1

        self.backbone_mapping = {b: node_to_idx[f"b_{b}"] for b in backbones}

        src_edges = []
        dst_edges = []
        edge_weights = []

        for row in bc.iter_rows(named=True):
            if row["country"] is None or row["backbone_id"] is None: continue
            u = node_to_idx[f"b_{row['backbone_id']}"]
            v = node_to_idx[f"c_{row['country']}"]
            src_edges.extend([u, v])
            dst_edges.extend([v, u])
            edge_weights.extend([row["weight"], row["weight"]])

        for row in bh.iter_rows(named=True):
            if row["host_order"] is None or row["backbone_id"] is None: continue
            u = node_to_idx[f"b_{row['backbone_id']}"]
            v = node_to_idx[f"h_{row['host_order']}"]
            src_edges.extend([u, v])
            dst_edges.extend([v, u])
            edge_weights.extend([row["weight"], row["weight"]])

        for row in cc.iter_rows(named=True):
            u = node_to_idx[f"c_{row['c1']}"]
            v = node_to_idx[f"c_{row['c2']}"]
            src_edges.extend([u])
            dst_edges.extend([v])
            edge_weights.extend([row["weight"]])

        num_nodes = idx
        edge_index = torch.tensor([src_edges, dst_edges], dtype=torch.long)
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)
        
        # Dummy node features (one-hot or identity if small, or random if large)
        # To save memory, we can use an Embedding layer or identity matrix
        x = torch.eye(num_nodes) if num_nodes < 2000 else torch.randn((num_nodes, 16))
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
        from torch_geometric.nn import GCNConv
        import torch.nn.functional as F

        class GNN(torch.nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels):
                super().__init__()
                self.conv1 = GCNConv(in_channels, hidden_channels)
                self.conv2 = GCNConv(hidden_channels, out_channels)

            def forward(self, x, edge_index, edge_weight):
                x = self.conv1(x, edge_index, edge_weight)
                x = F.relu(x)
                x = self.conv2(x, edge_index, edge_weight)
                return x

        model = GNN(x.size(1), 16, 8)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Improved link prediction loss with MarginRankingLoss proxy
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            z = model(data.x, data.edge_index, data.edge_attr)
            
            if edge_index.size(1) > 0:
                idx_sample = torch.randint(0, edge_index.size(1), (min(10000, edge_index.size(1)),))
                u = edge_index[0, idx_sample]
                v = edge_index[1, idx_sample]
                
                # Positive scores (cosine similarity proxy)
                pos_score = (z[u] * z[v]).sum(dim=-1)
                
                # Negative scores (harder negative sampling)
                neg_v = torch.randint(0, num_nodes, (u.size(0),))
                neg_score = (z[u] * z[neg_v]).sum(dim=-1)
                
                # Margin loss: push pos_score > neg_score + margin
                loss = torch.relu(1.0 - pos_score + neg_score).mean()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            final_embeddings = model(data.x, data.edge_index, data.edge_attr).numpy()
        
        self.embeddings = final_embeddings

    def transform(self, backbone_ids: pl.Series) -> pl.DataFrame:
        if self.embeddings is None:
            raise ValueError("Model not fitted")
        
        b_ids = backbone_ids.to_list()
        embeds = []
        d = self.embeddings.shape[1]
        for b in b_ids:
            if b in self.backbone_mapping:
                embeds.append(self.embeddings[self.backbone_mapping[b]])
            else:
                embeds.append(np.zeros(d))
                
        emb_array = np.array(embeds)
        cols = {f"gnn_embed_{i}": emb_array[:, i] for i in range(d)}
        cols["backbone_id"] = b_ids
        return pl.DataFrame(cols)

