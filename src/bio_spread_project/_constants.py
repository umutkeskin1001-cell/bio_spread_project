from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------
# Numerical safety & validation
# ---------------------------------------------------------------
EPSILON: float = 1e-12
LOG_EPSILON: float = 1e-6
MAX_YEAR: int = 2100
MIN_YEAR: int = 1900
MIN_KNOWN_RECORDS: int = 6
MIN_KNOWN_COUNTRIES: int = 2

# ---------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------
KNOWNESS_RECORDS_CAP: int = 8
KNOWNESS_COUNTRIES_CAP: int = 5
MIN_NEW_COUNTRIES_FOR_SPREAD: int = 2
SPLIT_YEAR_DEFAULT: int = 2020
HORIZON_YEARS_DEFAULT: int = 3

# ---------------------------------------------------------------
# Temporal validation
# ---------------------------------------------------------------
TEMPORAL_CALIB_FRACTION: float = 0.2
TEMPORAL_MIN_SPLITS: int = 3
TEMPORAL_HOLDOUT_FRACTION: float = 0.8

# ---------------------------------------------------------------
# Model training
# ---------------------------------------------------------------
CV_N_SPLITS_DEFAULT: int = 5
CV_RANDOM_STATE: int = 7
N_ESTIMATORS_RF: int = 200
MAX_DEPTH_RF: int | None = None
MIN_SAMPLES_LEAF_RF: int = 5
HGB_MAX_ITER: int = 200
HGB_MAX_DEPTH: int | None = 4
HGB_MIN_SAMPLES_LEAF: int = 20
HGB_L2_REG: float = 0.1
RIDGE_ALPHA: float = 0.1
LOGREG_C_DEFAULT: float = 0.5
LOGREG_MAX_ITER: int = 8000
LAMBDA_RANK_LEAVES: int = 31
LAMBDA_RANK_LR: float = 0.05
LAMBDA_RANK_ROUNDS: int = 200

# ---------------------------------------------------------------
# Evidential NN
# ---------------------------------------------------------------
EVID_HIDDEN_DIM: int = 64
EVID_DROPOUT: float = 0.3
EVID_EPOCHS: int = 20
EVID_LR: float = 1e-3
EVID_LAMBDA_CAL: float = 0.1
EVID_LOG_CLAMP: float = 20.0
EVID_N_CLASSES: int = 2

# ---------------------------------------------------------------
# Conformal prediction
# ---------------------------------------------------------------
CONFORMAL_ALPHA: float = 0.1
CONFORMAL_QHAT_DEFAULT: float = 0.5
RISK_CRC_TARGET_FPR: float = 0.05

# ---------------------------------------------------------------
# OOD detection
# ---------------------------------------------------------------
OOD_KNN_K: int = 5
OOD_REJECT_QUANTILE: float = 0.995

# ---------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------
FOCAL_GAMMA: float = 2.0
FOCAL_MAX_PAIRS: int = 4096
NDCG_TOPK: int = 25

# ---------------------------------------------------------------
# Meta ensemble
# ---------------------------------------------------------------
META_HIDDEN_DIM_BASIC: int = 32
META_HIDDEN_DIM_FOCAL: int = 64
META_BATCH_SIZE: int = 256
META_EPOCHS: int = 50
META_LR: float = 0.001
META_WD: float = 1e-4

# ---------------------------------------------------------------
# Ensemble mixture weights
# ---------------------------------------------------------------
RANKER_MIXTURE_WEIGHT: float = 0.1
EVIDENTIAL_MIXTURE_WEIGHT: float = 0.9

# ---------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------
CALIBRATION_BINS: int = 10
CALIBRATION_CV_SPLITS: int = 5

# ---------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------
BOOTSTRAP_N_RESAMPLES_DEFAULT: int = 1000
BOOTSTRAP_N_RESAMPLES_LIGHT: int = 200
BOOTSTRAP_CONFIDENCE: float = 0.95
BOOTSTRAP_SEED: int = 7
EXTERNAL_BOOTSTRAP_N_RESAMPLES: int = 200

# ---------------------------------------------------------------
# Model selection weights
# ---------------------------------------------------------------
SELECTION_WEIGHT_AUC: float = 0.40
SELECTION_WEIGHT_AP: float = 0.30
SELECTION_WEIGHT_CAL: float = 0.30

# ---------------------------------------------------------------
# Knownness derivation weights
# ---------------------------------------------------------------
KNOWNESS_W_DEPTH: float = 0.35
KNOWNESS_W_CONFIDENCE: float = 0.25
KNOWNESS_W_PURITY: float = 0.25
KNOWNESS_W_MISSING: float = 0.15

# ---------------------------------------------------------------
# Confidence tier thresholds
# ---------------------------------------------------------------
TIER_HIGH_KNOWNESS_MIN: float = 0.55
TIER_HIGH_PROB_UPPER: float = 0.75
TIER_HIGH_PROB_LOWER: float = 0.25
TIER_LOW_KNOWNESS_MAX: float = 0.25
TIER_AMBIGUOUS_LOWER: float = 0.40
TIER_AMBIGUOUS_UPPER: float = 0.60

TIER_GEO_HIGH_KNOWNESS_MIN: float = 0.60
TIER_GEO_HIGH_PROB_UPPER: float = 0.80
TIER_GEO_HIGH_PROB_LOWER: float = 0.20
TIER_GEO_MED_KNOWNESS_MIN: float = 0.40

# ---------------------------------------------------------------
# AMR priority weights
# ---------------------------------------------------------------
PRIORITY_AMR_WEIGHTS: dict[str, float] = {
    "blaNDM": 5.0,
    "blaKPC": 5.0,
    "blaOXA-48": 5.0,
    "blaVIM": 5.0,
    "blaIMP": 5.0,
    "mcr-1": 5.0,
    "blaCTX-M": 3.0,
    "vanA": 3.0,
    "vanB": 3.0,
}

# ---------------------------------------------------------------
# Taxonomy mapping
# ---------------------------------------------------------------
GENUS_TO_ORDER: dict[str, str] = {
    "Escherichia": "Enterobacterales",
    "Klebsiella": "Enterobacterales",
    "Salmonella": "Enterobacterales",
    "Enterobacter": "Enterobacterales",
    "Pseudomonas": "Pseudomonadales",
    "Acinetobacter": "Moraxellales",
    "Staphylococcus": "Bacillales",
    "Enterococcus": "Lactobacillales",
    "Vibrio": "Vibrionales",
    "Campylobacter": "Campylobacterales",
}

CLINICAL_TERMS: frozenset[str] = frozenset({"clinical", "hospital", "patient", "human"})

# ---------------------------------------------------------------
# Leakage detection
# ---------------------------------------------------------------
LEAKAGE_BLOCKLIST: tuple[str, ...] = (
    "spread_label",
    "n_new_countries",
    "n_new_macro_regions",
    "macro_region_jump_label",
    "future",
)
LEAKAGE_NAME_TOKENS: tuple[str, ...] = (
    "future",
    "test_",
    "label",
    "target",
    "outcome",
    "n_new_",
    "time_to_",
    "event_within_",
    "jump",
)
LEAKAGE_AUC_CEILING: float = 0.60
LEAKAGE_CANARY_AUC_FLOOR: float = 0.90
LEAKAGE_SINGLE_FEATURE_AUC_MAX: float = 0.85
LEAKAGE_N_PERMUTATIONS: int = 4
LEAKAGE_N_TRIALS: int = 8
LEAKAGE_TOP_K: int = 3

# ---------------------------------------------------------------
# I/O
# ---------------------------------------------------------------
CHUNK_SIZE: int = 1024 * 1024


# ---------------------------------------------------------------
# Protocols for duck-typed model interfaces
# ---------------------------------------------------------------
@runtime_checkable
class ProbabilisticModel(Protocol):
    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def fit(self, X: NDArray[np.float64], y: NDArray[np.int64]) -> Any: ...


@runtime_checkable
class DecisionModel(Protocol):
    def decision_function(self, X: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def fit(self, X: NDArray[np.float64], y: NDArray[np.int64]) -> Any: ...


@runtime_checkable
class Calibrator(Protocol):
    def transform(self, probs: NDArray[np.float64]) -> NDArray[np.float64]: ...
