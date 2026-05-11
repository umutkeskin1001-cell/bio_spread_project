import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix, precision_score, recall_score, f1_score
from typing import Dict, Any

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, uncertainty: np.ndarray) -> Dict[str, Any]:
    """
    Compute relevant metrics including uncertainty calibration and confusion matrix components.
    """
    metrics = {}
    y_pred = (y_prob > 0.5).astype(int)
    
    # Standard classification metrics
    if len(np.unique(y_true)) > 1:
        metrics['auc'] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics['auc'] = 0.5
        
    metrics['brier'] = float(brier_score_loss(y_true, y_prob))
    metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics['tn'] = int(tn)
    metrics['fp'] = int(fp)
    metrics['fn'] = int(fn)
    metrics['tp'] = int(tp)
    
    # Uncertainty metrics
    # High uncertainty should ideally correlate with incorrect predictions
    correct = y_pred == y_true
    if len(np.unique(correct)) > 1:
        metrics['uncertainty_auc'] = float(roc_auc_score(1 - correct.astype(int), uncertainty))
    else:
        metrics['uncertainty_auc'] = 0.5
        
    metrics['avg_uncertainty'] = float(np.mean(uncertainty))
    
    return metrics
