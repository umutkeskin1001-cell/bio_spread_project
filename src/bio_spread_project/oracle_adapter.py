import torch
import numpy as np
import polars as pl
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Union

class BioOracleDataset(Dataset):
    """
    Translates raw Polars data into the exact tensors expected by SovereignOracleNet.
    x_gene: padded sequences of gene indices
    t: temporal features
    y: binary labels (optional)
    """
    def __init__(self, df: pl.DataFrame, vocab: dict[str, int], max_len: int = 200, is_train: bool = True):
        self.df = df
        self.vocab = vocab
        self.max_len = max_len
        self.is_train = is_train
        
        # Build features
        self.x_gene, self.t, self.y = self._build_tensors()

    def _build_tensors(self) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # Extract features
        n_samples = len(self.df)
        
        # We assume the dataframe has a column 'amr_genes' (list of strings) or similar
        # For robustness, we will extract anything that looks like a gene family or AMR class
        # If the columns are dummy-encoded (e.g., 'amr_class_KPC': 1), we convert to indices.
        
        gene_cols = [c for c in self.df.columns if c.startswith("amr_") or c.startswith("inc_")]
        
        x_gene_list = []
        for i in range(n_samples):
            row = self.df.row(i, named=True)
            genes = []
            for col in gene_cols:
                val = row.get(col)
                if val is not None and val > 0:
                    if col in self.vocab:
                        genes.append(self.vocab[col])
                    else:
                        # Add to vocab if it's training
                        idx = len(self.vocab) + 1 # 0 is padding
                        self.vocab[col] = idx
                        genes.append(idx)
            
            # Truncate or pad
            if len(genes) > self.max_len:
                genes = genes[:self.max_len]
            else:
                genes = genes + [0] * (self.max_len - len(genes))
            x_gene_list.append(genes)
            
        x_gene = torch.tensor(x_gene_list, dtype=torch.long)
        
        # Extract time
        t_col = "year" if "year" in self.df.columns else "time_offset"
        if t_col in self.df.columns:
            t_vals = self.df[t_col].fill_null(0.0).to_numpy().astype(np.float32)
            t = torch.tensor(t_vals).unsqueeze(1) # [B, 1]
        else:
            t = torch.zeros((n_samples, 1), dtype=torch.float32)
            
        # Extract labels
        if self.is_train and "is_widely_spread" in self.df.columns:
            y_vals = self.df["is_widely_spread"].fill_null(0).to_numpy().astype(np.longlong)
            y = torch.tensor(y_vals, dtype=torch.long)
        else:
            y = None
            
        return x_gene, t, y

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, int]]:
        if self.y is not None:
            return self.x_gene[idx], self.t[idx], self.y[idx]
        return self.x_gene[idx], self.t[idx], -1

def create_oracle_dataloader(df: pl.DataFrame, vocab: dict[str, int], batch_size: int = 32, is_train: bool = True) -> DataLoader:
    dataset = BioOracleDataset(df, vocab, is_train=is_train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train)
