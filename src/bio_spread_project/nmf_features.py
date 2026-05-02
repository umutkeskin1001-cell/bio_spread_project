import numpy as np
import polars as pl
from sklearn.decomposition import NMF


def build_nmf_diffusion_features(records: pl.LazyFrame, split_year: int) -> pl.DataFrame:
    pre_obs = records.filter(pl.col("year") <= split_year).collect()

    if pre_obs.is_empty():
        return pl.DataFrame({"backbone_id": [], "reach_potential": [], "saturation_deficit": []})

    # Use mobility_score and sum as lengths/weights per backbone-country pair
    pivot_df = pre_obs.group_by(["backbone_id", "country"]).len()

    pivot_wide = pivot_df.pivot(values="len", index="backbone_id", columns="country").fill_null(0)

    backbones = pivot_wide["backbone_id"].to_numpy()
    countries = [c for c in pivot_wide.columns if c != "backbone_id"]
    B = pivot_wide.select(countries).to_numpy()

    n_components = min(4, B.shape[0], B.shape[1])
    if n_components < 1:
        return pl.DataFrame({
            "backbone_id": backbones,
            "reach_potential": np.zeros(len(backbones)),
            "saturation_deficit": np.zeros(len(backbones))
        })

    nmf = NMF(n_components=n_components, init='nndsvd' if min(B.shape) >= n_components else 'random', random_state=42)
    W = nmf.fit_transform(B)
    H = nmf.components_

    # Smooth H: country connectivity C proxy
    # C is Jaccard similarity between country observation profiles
    # C = (B^T * B) / (sum(B^T) + sum(B) - B^T * B)
    B_bin = (B > 0).astype(float)
    intersection = B_bin.T @ B_bin
    sz = B_bin.sum(axis=0)
    union = sz[:, None] + sz[None, :] - intersection
    union[union == 0] = 1.0
    C = intersection / union
    np.fill_diagonal(C, 1.0)

    alpha = 0.7
    H_smooth = alpha * H + (1 - alpha) * (C @ H.T).T

    Y_hat = W @ H_smooth

    threshold = 0.5
    current_counts = B_bin.sum(axis=1)
    predicted_counts = (Y_hat > threshold).sum(axis=1)

    reach_potential = predicted_counts - current_counts
    # clip reach potential to non-negative
    reach_potential = np.clip(reach_potential, 0, None)

    safe_pred = np.maximum(predicted_counts, 1)
    saturation_deficit = 1.0 - (current_counts / safe_pred)
    saturation_deficit = np.clip(saturation_deficit, 0.0, 1.0)

    return pl.DataFrame({
        "backbone_id": pl.Series(backbones),
        "reach_potential": pl.Series(reach_potential),
        "saturation_deficit": pl.Series(saturation_deficit)
    })
