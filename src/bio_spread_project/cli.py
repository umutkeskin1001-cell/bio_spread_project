import logging
import json
from pathlib import Path
from bio_spread_project.etl import SovereignETL
from bio_spread_project.train import run_training_cycle
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("OracleRebirth")

def main():
    logger.info("Initializing Sovereign Oracle v17 Rebirth...")
    
    # 1. ETL
    etl = SovereignETL()
    genetic_map = etl.load_genetic_map()
    
    inputs_dir = Path("data/project_inputs/geo_spread/inputs")
    train_df = etl.prepare_dataset(inputs_dir / "backbone_scored_train_only.tsv", genetic_map)
    test_df = etl.prepare_dataset(inputs_dir / "external_holdout_curated_v1.tsv", genetic_map)
    
    # 2. Train
    oof_probs, external_probs, vocab = run_training_cycle(train_df, test_df, genetic_map)
    
    # 3. Evaluate
    y_train = train_df["y"].to_list()
    y_test = test_df["y"].to_list()
    
    oof_auc = roc_auc_score(y_train, oof_probs)
    ext_auc = roc_auc_score(y_test, external_probs)
    
    # Optimal threshold from OOF
    p, r, t = precision_recall_curve(y_train, oof_probs)
    f1 = 2*p*r / (p+r+1e-8)
    best_t = t[np.argmax(f1)]
    
    ext_preds = (external_probs >= best_t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, ext_preds).ravel()
    
    results = {
        "OOF_AUC": float(oof_auc),
        "External_AUC": float(ext_auc),
        "External_Recall": float(tp/(tp+fn)),
        "External_Precision": float(tp/(tp+fp)),
        "FP": int(fp),
        "FN": int(fn),
        "Threshold": float(best_t)
    }
    
    logger.info(f"REBIRTH COMPLETE. RESULTS: {json.dumps(results, indent=4)}")
    
    Path("reports").mkdir(exist_ok=True)
    with open("reports/rebirth_v17_final.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
