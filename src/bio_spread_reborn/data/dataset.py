import torch
from torch.utils.data import Dataset
import polars as pl
import logging
from typing import Dict, List, Optional
from bio_spread_reborn.data.functional_tokenizer import FunctionalTokenizer

logger = logging.getLogger(__name__)

class BioDataset(Dataset):
    def __init__(self, 
                 df: pl.DataFrame, 
                 genetic_map: Dict[str, List[str]], 
                 max_genes: int = 50):
        """
        # and backcast features: n_countries_so_far, n_host_genera_so_far, years_since_first_obs, delta_countries_last_2y
        """
        self.df = df
        self.genetic_map = genetic_map
        self.max_genes = max_genes
        self.func_tokenizer = FunctionalTokenizer()
        
        # Cache for performance
        self.backbone_ids = df["backbone_id"].to_list()
        self.years = df["year"].to_list()
        self.labels = df["spread_label"].to_list() if "spread_label" in df.columns else None
        
        # Backcast features: any numeric column except year and label
        exclude = ["backbone_id", "year", "spread_label"]
        self.history_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
        self.history_features = df.select(self.history_cols).to_numpy().astype("float32")
        logger.info(f"BioDataset initialized with {len(self.history_cols)} numeric features: {self.history_cols}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        bid = self.backbone_ids[idx]
        year = self.years[idx]
        
        # 1. Genetic tokens (Raw)
        genes = self.genetic_map.get(bid, [])
        # We need a way to get tokens from genes here, 
        # but for simplicity we assume the caller provides a tokenized map or we reuse vocab.
        # Actually, let's assume we pass raw gene names and a separate tokenizer handles them.
        # For now, let's reuse the logic from previous session if possible.
        # Wait, I'll update the loader to provide tokenized genetic map.
        
        # 2. Functional features
        func_vec = self.func_tokenizer.encode(genes)
        
        # 3. History features
        hist_vec = self.history_features[idx]
        
        # 4. Label
        label = self.labels[idx] if self.labels is not None else 0
        
        # Placeholder for tokens (will be filled by loader/collator)
        return {
            "bid": bid,
            "year": year,
            "genes": genes, # Raw gene names for functional tokenizer
            "func_vec": torch.tensor(func_vec, dtype=torch.float32),
            "hist_vec": torch.tensor(hist_vec, dtype=torch.float32),
            "label": label
        }

def bio_collate_fn(batch, tokenizer, max_genes=50):
    """
    Custom collate to handle tokenization and padding.
    """
    bids = [item["bid"] for item in batch]
    years = torch.tensor([item["year"] for item in batch], dtype=torch.float32).unsqueeze(-1)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    
    func_vecs = torch.stack([item["func_vec"] for item in batch])
    hist_vecs = torch.stack([item["hist_vec"] for item in batch])
    
    # Tokenize raw genes
    token_list = []
    for item in batch:
        tokens = tokenizer.encode(item["genes"])[:max_genes]
        padding = [0] * (max_genes - len(tokens))
        token_list.append(torch.tensor(tokens + padding, dtype=torch.long))
        
    x_gene = torch.stack(token_list)
    
    return x_gene, func_vecs, hist_vecs, years, labels
