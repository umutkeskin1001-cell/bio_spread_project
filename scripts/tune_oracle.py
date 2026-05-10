import polars as pl
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Union

from bio_spread_project.oracle_core import SovereignOracleNet, evidential_loss

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("OracleTuner")

class RawDNAOracleDataset(Dataset):
    def __init__(self, data: list[dict], vocab: dict[str, int], max_len: int = 300, is_train: bool = True):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len
        self.is_train = is_train
        
        self.x_gene, self.t, self.y = self._build_tensors()

    def _build_tensors(self) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x_gene_list = []
        t_list = []
        y_list = []
        
        for item in self.data:
            genes = item["genes"]
            encoded_genes = []
            
            for g in genes:
                if g in self.vocab:
                    encoded_genes.append(self.vocab[g])
                elif self.is_train:
                    idx = len(self.vocab) + 1
                    self.vocab[g] = idx
                    encoded_genes.append(idx)
                    
            if len(encoded_genes) > self.max_len:
                encoded_genes = encoded_genes[:self.max_len]
            else:
                encoded_genes = encoded_genes + [0] * (self.max_len - len(encoded_genes))
                
            x_gene_list.append(encoded_genes)
            t_list.append(item["t"])
            y_list.append(item["y"])
            
        x_gene = torch.tensor(x_gene_list, dtype=torch.long)
        t = torch.tensor(t_list, dtype=torch.float32).unsqueeze(1)
        
        if y_list and y_list[0] is not None:
            y = torch.tensor(y_list, dtype=torch.long)
        else:
            y = None
            
        return x_gene, t, y

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, int]]:
        if self.y is not None:
            return self.x_gene[idx], self.t[idx], self.y[idx]
        return self.x_gene[idx], self.t[idx], -1

def prepare_data():
    logger.info("Loading Silver Databases...")
    silver_dir = Path("data/project_inputs/silver")
    inputs_dir = Path("data/project_inputs/geo_spread/inputs")
    
    df_backbones = pl.read_csv(silver_dir / "plasmid_backbones.tsv", separator="\t", null_values=[""])
    df_amr = pl.read_csv(silver_dir / "plasmid_amr_hits.tsv", separator="\t", null_values=[""])
    
    df_amr_grouped = df_amr.group_by("sequence_accession").agg(pl.col("gene_symbol").drop_nulls())
    df_joined = df_backbones.join(df_amr_grouped, on="sequence_accession", how="left")
    
    backbone_dict = {}
    for row in df_joined.iter_rows(named=True):
        bid = row["backbone_id"]
        if bid is None:
            continue
            
        if bid not in backbone_dict:
            backbone_dict[bid] = set()
            
        genes = row.get("gene_symbol")
        if genes:
            for g in genes:
                if g: backbone_dict[bid].add(f"AMR_{g}")
                
        reps = row.get("replicon_types")
        if reps and isinstance(reps, str):
            for r in reps.split(","):
                if r.strip(): backbone_dict[bid].add(f"REP_{r.strip()}")
                
    logger.info(f"Extracted genetic makeup for {len(backbone_dict)} unique backbones.")
    
    df_train = pl.read_csv(inputs_dir / "backbone_scored_train_only.tsv", separator="\t")
    df_test = pl.read_csv(inputs_dir / "external_holdout_curated_v1.tsv", separator="\t")
    
    def df_to_data_list(df: pl.DataFrame) -> list[dict]:
        data = []
        for row in df.iter_rows(named=True):
            bid = row.get("backbone_id")
            if not bid: continue
            genes = list(backbone_dict.get(bid, []))
            y = row.get("spread_label")
            if y is None: y = 0.0
            t = row.get("T_eff_norm")
            if t is None: t = 0.0
            data.append({"backbone_id": bid, "genes": genes, "t": float(t), "y": int(y)})
        return data

    train_data = df_to_data_list(df_train)
    test_data = df_to_data_list(df_test)
    
    logger.info(f"Train samples: {len(train_data)}")
    logger.info(f"Test samples: {len(test_data)}")
    
    return train_data, test_data

def tune_oracle():
    train_data, test_data = prepare_data()
    
    y_train = [d["y"] for d in train_data]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_probs = np.zeros(len(train_data))
    oof_targets = np.zeros(len(train_data))
    external_preds_ensemble = np.zeros(len(test_data))
    
    logger.info("Starting 5-Fold Stratified CV OOF Evaluation")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_data, y_train)):
        logger.info(f"--- FOLD {fold+1}/5 ---")
        
        fold_train_data = [train_data[i] for i in train_idx]
        fold_val_data = [train_data[i] for i in val_idx]
        
        vocab = {}
        train_dataset = RawDNAOracleDataset(fold_train_data, vocab, max_len=300, is_train=True)
        val_dataset = RawDNAOracleDataset(fold_val_data, vocab, max_len=300, is_train=False)
        test_dataset = RawDNAOracleDataset(test_data, vocab, max_len=300, is_train=False)
        
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        vocab_size = max(2, len(vocab) + 1)
        
        model = SovereignOracleNet(vocab_size=vocab_size, h_dim=256, d_dim=64)
        optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-3)
        epochs = 40
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for x_gene, t, y in train_loader:
                optimizer.zero_grad()
                prob, unc, alpha = model(x_gene, t)
                loss = evidential_loss(alpha, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            
        # OOF Prediction
        model.eval()
        fold_val_probs = []
        with torch.no_grad():
            for x_gene, t, y in val_loader:
                prob, unc, alpha = model(x_gene, t)
                fold_val_probs.extend(prob.cpu().numpy())
                
        oof_probs[val_idx] = fold_val_probs
        oof_targets[val_idx] = [fold_val_data[i]["y"] for i in range(len(fold_val_data))]
        
        fold_auc = roc_auc_score(oof_targets[val_idx], fold_val_probs)
        logger.info(f"Fold {fold+1} Val AUC: {fold_auc:.4f}")
        
        # External Prediction
        fold_ext_probs = []
        with torch.no_grad():
            for x_gene, t, y in test_loader:
                prob, unc, alpha = model(x_gene, t)
                fold_ext_probs.extend(prob.cpu().numpy())
                
        external_preds_ensemble += np.array(fold_ext_probs) / 5.0

    try:
        oof_auc = roc_auc_score(oof_targets, oof_probs)
        oof_ap = average_precision_score(oof_targets, oof_probs)
        
        # Calculate Optimal Threshold via OOF F1-Score
        precisions, recalls, thresholds = precision_recall_curve(oof_targets, oof_probs)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = float(thresholds[optimal_idx]) if optimal_idx < len(thresholds) else 0.5
        logger.info(f"Calculated Optimal Probability Threshold: {optimal_threshold:.4f}")
        
        oof_preds_binary = [1 if p >= optimal_threshold else 0 for p in oof_probs]
        o_tn, o_fp, o_fn, o_tp = confusion_matrix(oof_targets, oof_preds_binary).ravel()
        
        ext_targets = [d["y"] for d in test_data]
        ext_auc = roc_auc_score(ext_targets, external_preds_ensemble)
        ext_ap = average_precision_score(ext_targets, external_preds_ensemble)
        ext_preds_binary = [1 if p >= optimal_threshold else 0 for p in external_preds_ensemble]
        e_tn, e_fp, e_fn, e_tp = confusion_matrix(ext_targets, ext_preds_binary).ravel()
        
        metrics = {
            "OOF_ROC_AUC": float(oof_auc),
            "OOF_Average_Precision": float(oof_ap),
            "OOF_Recall": float(o_tp / (o_tp + o_fn)) if (o_tp + o_fn) > 0 else 0.0,
            "OOF_Precision": float(o_tp / (o_tp + o_fp)) if (o_tp + o_fp) > 0 else 0.0,
            "External_ROC_AUC": float(ext_auc),
            "External_Average_Precision": float(ext_ap),
            "External_Recall": float(e_tp / (e_tp + e_fn)) if (e_tp + e_fn) > 0 else 0.0,
            "External_Precision": float(e_tp / (e_tp + e_fp)) if (e_tp + e_fp) > 0 else 0.0,
            "External_False_Positives": int(e_fp),
            "External_False_Negatives": int(e_fn)
        }
        
        logger.info("\n" + "="*50)
        logger.info("FINAL TUNED ORACLE V15 METRICS (OOF vs EXTERNAL)")
        logger.info("="*50)
        logger.info(f"--- OUT-OF-FOLD (OOF) VALIDATION ---")
        logger.info(f"ROC AUC           : {metrics['OOF_ROC_AUC']:.4f}")
        logger.info(f"Average Precision : {metrics['OOF_Average_Precision']:.4f}")
        logger.info(f"Recall            : {metrics['OOF_Recall']:.4f}")
        logger.info(f"Precision         : {metrics['OOF_Precision']:.4f}")
        logger.info(f"--- EXTERNAL HOLDOUT ENSEMBLE ---")
        logger.info(f"ROC AUC           : {metrics['External_ROC_AUC']:.4f}")
        logger.info(f"Average Precision : {metrics['External_Average_Precision']:.4f}")
        logger.info(f"Recall            : {metrics['External_Recall']:.4f}")
        logger.info(f"Precision         : {metrics['External_Precision']:.4f}")
        logger.info(f"False Positives   : {metrics['External_False_Positives']}")
        logger.info(f"False Negatives   : {metrics['External_False_Negatives']}")
        logger.info("="*50)
        
        with open("reports/oracle_v15_tuned_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
            
    except Exception as e:
        logger.error(f"Error calculating final metrics: {e}")

if __name__ == "__main__":
    tune_oracle()
