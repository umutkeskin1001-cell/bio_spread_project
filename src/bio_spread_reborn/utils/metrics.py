import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix, precision_score, recall_score, f1_score
from typing import Dict, Any

def calculate_metrics(y_true, y_prob, uncertainty) -> Dict[str, Any]:
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    uncertainty = np.array(uncertainty)
    y_pred = (y_prob > 0.5).astype(int)
    
    metrics = {}
    if len(np.unique(y_true)) > 1:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics['roc_auc'] = 0.5
        
    metrics['brier_score'] = float(brier_score_loss(y_true, y_prob))
    metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics['f1_score'] = float(f1_score(y_true, y_pred, zero_division=0))
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics['tn'] = int(tn)
    metrics['fp'] = int(fp)
    metrics['fn'] = int(fn)
    metrics['tp'] = int(tp)
    
    correct = y_pred == y_true
    if len(np.unique(correct)) > 1:
        metrics['uncertainty_auc'] = float(roc_auc_score(1 - correct.astype(int), uncertainty))
    else:
        metrics['uncertainty_auc'] = 0.5
        
    metrics['avg_uncertainty'] = float(np.mean(uncertainty))
    return metrics

def print_report(metrics: Dict[str, Any]):
    print("\n" + "="*40)
    print("       BIO-SPREAD EVALUATION REPORT")
    print("="*40)
    print(f"ROC AUC:          {metrics['roc_auc']:.4f}")
    print(f"Brier Score:      {metrics['brier_score']:.4f}")
    print(f"Precision:        {metrics['precision']:.4f}")
    print(f"Recall:           {metrics['recall']:.4f}")
    print(f"F1 Score:         {metrics['f1_score']:.4f}")
    print("-" * 40)
    print(f"TP/FP/TN/FN:      {metrics['tp']}/{metrics['fp']}/{metrics['tn']}/{metrics['fn']}")
    print(f"Uncertainty AUC:  {metrics['uncertainty_auc']:.4f}")
    print(f"Avg Uncertainty:  {metrics['avg_uncertainty']:.4f}")
    print("="*40 + "\n")
