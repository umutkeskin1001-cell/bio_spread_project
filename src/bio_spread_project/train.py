import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
import numpy as np
import logging
import polars as pl

from bio_spread_project.model import SovereignOracleNet, evidential_loss

logger = logging.getLogger("Train")

class DNADataset(Dataset):
    def __init__(self, df, vocab, max_len=300):
        self.max_len = max_len
        self.y = torch.tensor(df["y"].to_list(), dtype=torch.long)
        self.t = torch.tensor(df["t"].to_list(), dtype=torch.float32).unsqueeze(1)
        
        # Encode genes
        encoded = []
        for genes in df["genes"]:
            seq = [vocab.get(g, 0) for g in genes][:max_len]
            seq = (seq + [0] * max_len)[:max_len]
            encoded.append(seq)
        self.x = torch.tensor(encoded, dtype=torch.long)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.t[idx], self.y[idx]

def run_training_cycle(train_df, test_df, genetic_map):
    # Build Vocab
    unique_genes = sorted(list(set([g for genes in train_df["genes"] for g in genes])))
    vocab = {g: i+1 for i, g in enumerate(unique_genes)}
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_labels = train_df["y"].to_list()
    
    oof_probs = np.zeros(len(train_df))
    external_preds = np.zeros(len(test_df))
    
    logger.info("Executing 5-Fold Stratified Cross-Validation...")
    
    # Add index for Polars filtering
    train_df = train_df.with_columns(pl.Series(np.arange(len(train_df))).alias("_idx"))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y_labels)):
        logger.info(f"Fold {fold+1}/5 Training...")
        
        # Polars filtering by row index
        train_ds_df = train_df.filter(pl.col("_idx").is_in(train_idx))
        val_ds_df = train_df.filter(pl.col("_idx").is_in(val_idx))
        
        train_ds = DNADataset(train_ds_df, vocab)
        val_ds = DNADataset(val_ds_df, vocab)
        test_ds = DNADataset(test_df, vocab)
        
        loader = DataLoader(train_ds, batch_size=128, shuffle=True)
        model = SovereignOracleNet(vocab_size=len(vocab)+1)
        optimizer = optim.AdamW(model.parameters(), lr=0.005)
        
        for epoch in range(30):
            model.train()
            for x, t, y in loader:
                optimizer.zero_grad()
                p, alpha = model(x, t)
                loss = evidential_loss(alpha, y)
                loss.backward()
                optimizer.step()
        
        model.eval()
        with torch.no_grad():
            p_val, _ = model(val_ds.x, val_ds.t)
            oof_probs[val_idx] = p_val.numpy()
            
            p_test, _ = model(test_ds.x, test_ds.t)
            external_preds += p_test.numpy() / 5.0
            
    return oof_probs, external_preds, vocab
