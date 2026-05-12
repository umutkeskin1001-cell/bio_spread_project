import torch
from torch.utils.data import Dataset
import polars as pl
import logging
import numpy as np
from typing import Dict, List, Optional
from bio_spread_reborn.data.functional_tokenizer import FunctionalTokenizer

logger = logging.getLogger(__name__)

class BioDataset(Dataset):
    def __init__(self, 
                 df: pl.DataFrame, 
                 genetic_map: Dict[str, List[str]], 
                 extra_features: Optional[Dict[str, np.ndarray]] = None,
                 max_genes: int = 50,
                 return_pairs: bool = False):
        """
        # and backcast features: n_countries_so_far, n_host_genera_so_far, years_since_first_obs, delta_countries_last_2y
        """
        self.df = df
        self.genetic_map = genetic_map
        self.extra_features = extra_features
        self.max_genes = max_genes
        self.func_tokenizer = FunctionalTokenizer()
        self.return_pairs = return_pairs
        
        # Cache for performance
        self.backbone_ids = df["backbone_id"].to_list()
        self.years = df["year"].to_list()
        self.labels = df["spread_label"].to_list() if "spread_label" in df.columns else None
        
        # Precompute pairs for Ranking Loss
        self.pairs = {}
        if return_pairs:
            from collections import defaultdict
            import random
            bid_to_idx = defaultdict(list)
            for i, bid in enumerate(self.backbone_ids):
                bid_to_idx[bid].append(i)
            for i, bid in enumerate(self.backbone_ids):
                candidates = [j for j in bid_to_idx[bid] if j != i]
                if candidates:
                    self.pairs[i] = random.choice(candidates)
        
        # Backcast features: any numeric column except year and label
        exclude = ["backbone_id", "year", "spread_label"]
        self.history_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
        self.history_features = df.select(self.history_cols).to_numpy().astype("float32")
        # Scale history features (counts, years) using log1p to prevent vanishing/exploding gradients
        self.history_features = np.log1p(np.maximum(self.history_features, 0.0))
        logger.info(f"BioDataset initialized with {len(self.history_cols)} numeric features: {self.history_cols}. Pairs: {return_pairs}")



    def __len__(self):
        return len(self.df)

    def _get_single_item(self, idx):
        bid = self.backbone_ids[idx]
        year = self.years[idx]
        
        # 1. Genetic tokens (Raw)
        genes = self.genetic_map.get(bid, [])
        
        # 2. Functional features
        func_vec = self.func_tokenizer.encode(genes)
        
        # 3. History features
        hist_vec = self.history_features[idx]
        
        # 4. Extra features (PFP + EBV + CAV) using snapshot-aware key
        key = f"{bid}_{year}"
        if self.extra_features and key in self.extra_features:
            extra_vec = self.extra_features[key]
        elif self.extra_features and bid in self.extra_features:
            # Fallback for old cached data if still present
            extra_vec = self.extra_features[bid]
        else:
            # Fallback to zeros (16+6+8 = 30)
            extra_vec = np.zeros(30, dtype=np.float32)
            
        # 5. Label
        label = self.labels[idx] if self.labels is not None else 0
        
        return {
            "bid": bid,
            "year": year,
            "genes": genes,
            "func_vec": torch.tensor(func_vec, dtype=torch.float32),
            "hist_vec": torch.tensor(hist_vec, dtype=torch.float32),
            "extra_vec": torch.tensor(extra_vec, dtype=torch.float32),
            "label": label
        }

    def __getitem__(self, idx):
        item = self._get_single_item(idx)
        if self.return_pairs and idx in self.pairs:
            pair_item = self._get_single_item(self.pairs[idx])
            return [item, pair_item]
        return [item]


def bio_collate_fn(batch, tokenizer, max_genes=50):
    """
    Custom collate to handle tokenization and padding.
    """
    flat_batch = [item for sublist in batch for item in sublist]
    
    bids = [item["bid"] for item in flat_batch]
    # TimeGate Normalization: (year - 2000) / 30.0
    years = torch.tensor([(item["year"] - 2000.0) / 30.0 for item in flat_batch], dtype=torch.float32).unsqueeze(-1)
    labels = torch.tensor([item["label"] for item in flat_batch], dtype=torch.long)
    
    func_vecs = torch.stack([item["func_vec"] for item in flat_batch])
    hist_vecs = torch.stack([item["hist_vec"] for item in flat_batch])
    extra_vecs = torch.stack([item["extra_vec"] for item in flat_batch])
    
    # Tokenize raw genes
    token_list = []
    for item in flat_batch:
        tokens = tokenizer.encode(item["genes"])[:max_genes]
        padding = [0] * (max_genes - len(tokens))
        token_list.append(torch.tensor(tokens + padding, dtype=torch.long))
        
    x_gene = torch.stack(token_list)
    
    # Also pass raw years just in case
    raw_years = [item["year"] for item in flat_batch]
    
    return {
        "x_gene": x_gene,
        "x_func": func_vecs,
        "x_hist": hist_vecs,
        "x_extra": extra_vecs,
        "t": years,
        "y": labels,
        "bids": bids,
        "raw_years": raw_years
    }

