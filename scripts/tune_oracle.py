import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
import json
import logging
import numpy as np
from pathlib import Path

from bio_spread_project.oracle_core import SovereignOracleNet, evidential_loss
from bio_spread_project.data_engine import DataEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("OracleRebirth")

class RawDNAOracleDataset(Dataset):
    def __init__(self, data: list[dict], vocab: dict[str, int], max_len: int = 300, is_train: bool = True):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len
        self.is_train = is_train
        self.x_gene, self.t, self.y = self._build_tensors()

    def _build_tensors(self):
        x_gene_list, t_list, y_list = [], [], []
        for item in self.data:
            encoded_genes = []
            for g in item["genes"]:
                if g in self.vocab:
                    encoded_genes.append(self.vocab[g])
                elif self.is_train:
                    idx = len(self.vocab) + 1
                    self.vocab[g] = idx
                    encoded_genes.append(idx)
            
            encoded_genes = (encoded_genes[:self.max_len] + [0] * self.max_len)[:self.max_len]
            x_gene_list.append(encoded_genes)
            t_list.append(item["t"])
            y_list.append(item["y"])
            
        return torch.tensor(x_gene_list, dtype=torch.long), \
               torch.tensor(t_list, dtype=torch.float32).unsqueeze(1), \
               torch.tensor(y_list, dtype=torch.long)

    def __len__(self) -> int: return len(self.data)
    def __getitem__(self, idx: int): return self.x_gene[idx], self.t[idx], self.y[idx]

def tune_oracle():
    engine = DataEngine()
    backbone_dict = engine.scan_genetic_makeup()
    
    inputs_dir = Path("data/project_inputs/geo_spread/inputs")
    train_data = engine.load_scored_dataset(inputs_dir / "backbone_scored_train_only.tsv", backbone_dict)
    test_data = engine.load_scored_dataset(inputs_dir / "external_holdout_curated_v1.tsv", backbone_dict)
    
    y_train = [d["y"] for d in train_data]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_probs, oof_targets = np.zeros(len(train_data)), np.zeros(len(train_data))
    external_preds_ensemble = np.zeros(len(test_data))
    
    logger.info("Starting Sovereign Oracle v16 Rebirth Training (5-Fold CV)")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_data, y_train)):
        logger.info(f"--- FOLD {fold+1}/5 ---")
        vocab = {}
        train_ds = RawDNAOracleDataset([train_data[i] for i in train_idx], vocab, is_train=True)
        val_ds = RawDNAOracleDataset([train_data[i] for i in val_idx], vocab, is_train=False)
        test_ds = RawDNAOracleDataset(test_data, vocab, is_train=False)
        
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
        
        model = SovereignOracleNet(vocab_size=len(vocab)+1)
        optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
        
        for epoch in range(40):
            model.train()
            for x, t, y in train_loader:
                optimizer.zero_grad()
                prob, unc, alpha = model(x, t)
                loss = evidential_loss(alpha, y)
                loss.backward()
                optimizer.step()
            scheduler.step()
            
        model.eval()
        with torch.no_grad():
            fold_probs = []
            for x, t, y in val_loader:
                p, u, a = model(x, t)
                fold_probs.extend(p.cpu().numpy())
            oof_probs[val_idx] = fold_probs
            oof_targets[val_idx] = [train_data[i]["y"] for i in val_idx]
            
            ext_probs = []
            for x, t, y in test_loader:
                p, u, a = model(x, t)
                ext_probs.extend(p.cpu().numpy())
            external_preds_ensemble += np.array(ext_probs) / 5.0

    # Optimal Threshold & Final Metrics
    precisions, recalls, thresholds = precision_recall_curve(oof_targets, oof_probs)
    f1 = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    opt_thresh = float(thresholds[np.argmax(f1)])
    
    ext_targets = [d["y"] for d in test_data]
    ext_auc = roc_auc_score(ext_targets, external_preds_ensemble)
    ext_preds = [1 if p >= opt_thresh else 0 for p in external_preds_ensemble]
    tn, fp, fn, tp = confusion_matrix(ext_targets, ext_preds).ravel()
    
    metrics = {
        "OOF_ROC_AUC": float(roc_auc_score(oof_targets, oof_probs)),
        "External_ROC_AUC": float(ext_auc),
        "External_Recall": float(tp/(tp+fn)),
        "External_Precision": float(tp/(tp+fp)),
        "FP": int(fp), "FN": int(fn), "Threshold": opt_thresh
    }
    logger.info(f"FINAL METRICS: {json.dumps(metrics, indent=4)}")
    with open("reports/oracle_v16_rebirth_metrics.json", "w") as f: json.dump(metrics, f)

if __name__ == "__main__": tune_oracle()
