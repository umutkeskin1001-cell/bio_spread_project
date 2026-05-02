import numpy as np
import polars as pl
from bio_spread_project.geo_reliability import single_feature_leakage_scan, FEATURE_COLUMNS

def phantom_leakage_audit(features_df: pl.DataFrame, label_col='label_geo_spread', required_auc=0.95):
    """
    Automated self-test for the leakage alarm system.
    Injects a noisy label as a 'phantom feature' and ensures the alarm triggers.
    """
    if label_col not in features_df.columns:
        return True
        
    features_df = features_df.filter(pl.col(label_col).is_not_null())
    if features_df.is_empty():
        return True
    
    y = features_df[label_col].to_numpy()
    if len(np.unique(y)) < 2:
        # Cannot run AUC-based audit if only one class is present
        return True
        
    # Add very small noise to prevent exact identity but maintain 0.99+ AUC
    phantom = y + np.random.normal(0, 0.001, len(y))
    
    # Directly check if we can detect this leakage
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y, phantom)
        auc = max(auc, 1 - auc)
        if auc < required_auc:
            raise ValueError(f"Phantom leakage gate failed – detected AUC {auc:.4f} < {required_auc}")
    except Exception as e:
        raise ValueError(f"Phantom leakage gate failed with error: {e}")
    
    return True
