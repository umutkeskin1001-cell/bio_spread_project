from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


class PhyloSpatialGraphEmbedder:
    def __init__(self) -> None:
        self.backbone_map: dict[str, int] = {}
        self.embeddings: np.ndarray | None = None

    def fit_transform(
        self,
        records: pl.DataFrame,
        *,
        split_year: int,
        backbone_ids: list[str],
        feature_matrix: np.ndarray,
        mash_path: Path | None = None,
        save_path: Path | None = None,
    ) -> pl.DataFrame:
        import torch
        import torch.nn.functional as F
        from torch_geometric.data import Data
        from torch_geometric.nn import GCNConv

        self.backbone_map = {bb: i for i, bb in enumerate(backbone_ids)}
        n_backbones = len(backbone_ids)
        if n_backbones == 0:
            return pl.DataFrame({"backbone_id": []})

        pre = records.filter(pl.col("year") <= split_year)
        edges: list[tuple[int, int, float]] = []

        if not pre.is_empty() and {"backbone_id", "country"}.issubset(pre.columns):
            bc = pre.select(["backbone_id", "country"]).drop_nulls().unique().group_by("country").agg(pl.col("backbone_id"))
            for row in bc.to_dicts():
                bbs = [b for b in row["backbone_id"] if b in self.backbone_map]
                for i in range(len(bbs)):
                    for j in range(i + 1, len(bbs)):
                        u = self.backbone_map[bbs[i]]
                        v = self.backbone_map[bbs[j]]
                        edges.append((u, v, 1.0))

        if not pre.is_empty() and {"backbone_id", "host_genus"}.issubset(pre.columns):
            bh = pre.select(["backbone_id", "host_genus"]).drop_nulls().unique().group_by("host_genus").agg(pl.col("backbone_id"))
            for row in bh.to_dicts():
                bbs = [b for b in row["backbone_id"] if b in self.backbone_map]
                for i in range(len(bbs)):
                    for j in range(i + 1, len(bbs)):
                        u = self.backbone_map[bbs[i]]
                        v = self.backbone_map[bbs[j]]
                        edges.append((u, v, 0.8))

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
            # disconnected fallback
            self.embeddings = np.zeros((n_backbones, 8), dtype=np.float32)
            return self._to_df(backbone_ids)

        src = []
        dst = []
        wts = []
        for u, v, w in edges:
            src.extend([u, v])
            dst.extend([v, u])
            wts.extend([w, w])

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(wts, dtype=torch.float32)
        x = torch.tensor(feature_matrix[:, : min(feature_matrix.shape[1], 32)], dtype=torch.float32)
        if x.shape[1] < 32:
            pad = torch.zeros((x.shape[0], 32 - x.shape[1]), dtype=torch.float32)
            x = torch.cat([x, pad], dim=1)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        class GCN(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = GCNConv(32, 16)
                self.conv2 = GCNConv(16, 8)

            def forward(self, x_t: torch.Tensor, e_idx: torch.Tensor, e_w: torch.Tensor) -> torch.Tensor:
                h = self.conv1(x_t, e_idx, e_w)
                h = F.relu(h)
                return self.conv2(h, e_idx, e_w)

        model = GCN()
        opt = torch.optim.Adam(model.parameters(), lr=0.01)

        model.train()
        for _ in range(30):
            opt.zero_grad()
            z = model(data.x, data.edge_index, data.edge_attr)
            idx = torch.randint(0, data.edge_index.shape[1], (min(5000, data.edge_index.shape[1]),))
            u = data.edge_index[0, idx]
            v = data.edge_index[1, idx]
            pos = (z[u] * z[v]).sum(dim=-1)
            neg_v = torch.randint(0, n_backbones, (u.shape[0],))
            neg = (z[u] * z[neg_v]).sum(dim=-1)
            loss = torch.relu(1.0 - pos + neg).mean()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            self.embeddings = model(data.x, data.edge_index, data.edge_attr).cpu().numpy().astype(np.float32)

        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"backbone_map": self.backbone_map, "embeddings": self.embeddings}, save_path)

        return self._to_df(backbone_ids)

    def _to_df(self, backbone_ids: list[str]) -> pl.DataFrame:
        if self.embeddings is None:
            raise ValueError("embeddings are not fitted")
        cols = {f"psge_{i}": self.embeddings[:, i] for i in range(self.embeddings.shape[1])}
        cols["backbone_id"] = backbone_ids
        return pl.DataFrame(cols)
