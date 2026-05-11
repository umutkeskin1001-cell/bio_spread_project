import torch
from torch.utils.data import Dataset
import polars as pl
from typing import Dict, List
from bio_spread_reborn.data.tokenizer import GeneTokenizer

class BioDataset(Dataset):
    def __init__(self, 
                 df: pl.DataFrame, 
                 genetic_map: Dict[str, List[int]], 
                 target_col: str = 'spread_label'):
        """
        PyTorch Dataset for BioSpread.
        
        Args:
            df: DataFrame containing backbone_id, year, and target.
            genetic_map: Mapping from backbone_id to list of encoded gene tokens.
            target_col: Name of the target column.
        """
        self.df = df
        self.genetic_map = genetic_map
        self.target_col = target_col
        
        # Convert to tensors or lists for faster access
        self.backbone_ids = df['backbone_id'].to_list()
        self.years = torch.tensor(df['year'].to_numpy(), dtype=torch.float32).unsqueeze(-1)
        
        if target_col in df.columns:
            self.targets = torch.tensor(df[target_col].to_numpy(), dtype=torch.long)
        else:
            self.targets = None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        bid = self.backbone_ids[idx]
        # Get encoded genes from map
        # Default to all zeros (padding) if not found
        x_gene = torch.tensor(self.genetic_map.get(bid, [0] * 300), dtype=torch.long)
        t = self.years[idx]
        
        if self.targets is not None:
            y = self.targets[idx]
            return x_gene, t, y
        else:
            return x_gene, t
