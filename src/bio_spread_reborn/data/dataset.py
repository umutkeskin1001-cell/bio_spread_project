import torch
from torch.utils.data import Dataset
import polars as pl
from typing import Dict, List, Optional

class BioDataset(Dataset):
    def __init__(self, 
                 df: pl.DataFrame, 
                 genetic_map: Dict[str, List[int]], 
                 target_col: str = 'spread_label',
                 max_genes: int = 300):
        """
        PyTorch Dataset for BioSpread.
        """
        self.df = df
        self.genetic_map = genetic_map
        self.target_col = target_col
        self.max_genes = max_genes
        
        # Convert to lists/tensors for faster access
        self.backbone_ids = df['backbone_id'].to_list()
        self.years = torch.tensor(df['year'].to_numpy(), dtype=torch.float32).unsqueeze(-1)
        
        if target_col in df.columns:
            # Handle potential nulls in labels if they still exist
            targets = df[target_col].fill_null(0).to_numpy()
            self.targets = torch.tensor(targets, dtype=torch.long)
        else:
            self.targets = None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        bid = self.backbone_ids[idx]
        # Get encoded genes from map
        # Use a zero-padding fallback of the correct length
        tokens = self.genetic_map.get(bid)
        if tokens is None:
            tokens = [0] * self.max_genes
            
        x_gene = torch.tensor(tokens, dtype=torch.long)
        t = self.years[idx]
        
        if self.targets is not None:
            y = self.targets[idx]
            return x_gene, t, y
        else:
            return x_gene, t
