import polars as pl
import numpy as np
from sklearn.metrics import roc_auc_score

df = pl.read_csv("reports/run/predictions.csv") # Wait, predictions.csv might not have all features.
# Let's use the audit metadata or just run a script on the features.

# Actually, I'll use the feature surface directly.
paths = {
    "geo_spread_features": "data/project_inputs/geo_spread/inputs/backbone_scored.tsv"
}
df = pl.read_csv(paths["geo_spread_features"], separator="\t")

# Replicate the label logic from features.py
# (Wait, I should check if n_new_countries is already there)
label = df["spread_label"].to_numpy()

FEATURE_COLUMNS = [
    "T_eff_norm", "H_obs_specialization_norm", "A_eff_norm", "coherence_score",
    "backbone_purity_norm", "assignment_confidence_norm", "mash_neighbor_distance_train_norm",
    "orit_support", "H_external_host_range_norm", "geo_country_entropy_train",
    "geo_macro_region_entropy_train", "geo_dominant_region_share_train", "geo_country_record_count_train",
    "surv_intensity", "host_sampling_shannon", "reach_potential", "saturation_deficit"
]

print("Feature AUC Scan:")
for col in df.columns:
    if col in ["spread_label", "backbone_id", "country"]: continue
    try:
        val = df[col].fill_null(0.0).to_numpy()
        if len(np.unique(val)) < 2: continue
        auc = roc_auc_score(label, val)
        auc = max(auc, 1.0 - auc)
        if auc > 0.8:
            print(f"{col}: {auc:.4f}")
    except:
        pass
