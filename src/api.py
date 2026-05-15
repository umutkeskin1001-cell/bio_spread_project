"""
Production API for BioSpread Sovereign-X Pro.

Serves predictions from a trained SovereignX model.
Accepts backbone snapshot features, returns hazard probabilities for years 1-3.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from bio_spread_reborn.models.components import PlattScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BioSpreadAPI")

# ---------------------------------------------------------------------------
# Configuration (from environment)
# ---------------------------------------------------------------------------
CONFIG_PATH = os.getenv("CONFIG_PATH", "config/default.yaml")
FEATURE_DIR = os.getenv("FEATURE_DIR", "data/sovereign_features")
MODEL_PATH = os.getenv("MODEL_PATH", "best_model.pt")
PLATT_PATH = os.getenv("PLATT_PATH", "")
DEVICE = os.getenv("DEVICE", "cpu")

app = FastAPI(title="BioSpread Sovereign-X Pro")

# Lazy-loaded model artifacts
model: Optional[torch.nn.Module] = None  # SovereignX instance
platt_scaler: Optional[PlattScaler] = None
cfg = None
feature_dims = None
_norm_cache: dict[str, np.ndarray] = {}  # normalizer cache (loaded once at startup)


class SnapshotFeatures(BaseModel):
    """One timestep of snapshot features."""

    n_countries: float = 0.0
    n_hosts: float = 0.0
    years_since_first: float = 0.0
    new_countries_recent: float = 0.0
    new_countries_2y_ago: float = 0.0
    n_records: float = 0.0
    acceleration: float = 0.0
    expansion_ratio: float = 1.0
    niche_breadth: float = 0.0


class StaticFeatures(BaseModel):
    """Backbone-level static features."""

    log_size: float = 0.0
    gc: float = 0.5
    n_replicon_types: float = 0.0
    n_relaxase_types: float = 0.0
    mobility_score: float = 0.0
    is_conjugative: float = 0.0
    is_mobilizable: float = 0.0
    topology: float = 0.0
    n_orit_types: float = 0.0
    host_range_rank: float = 0.0


class TaxonomyIndices(BaseModel):
    """Optional taxonomy indices [phylum, class, order, family, genus]."""

    phylum_idx: int = 0
    class_idx: int = 0
    order_idx: int = 0
    family_idx: int = 0
    genus_idx: int = 0


class PredictRequest(BaseModel):
    """Prediction request for Sovereign-X Pro.

    Accepts a sequence of snapshot features + static backbone features.
    """

    snapshots: List[SnapshotFeatures]
    static: StaticFeatures
    taxonomy: Optional[TaxonomyIndices] = None


class PredictResponse(BaseModel):
    """Sovereign-X Pro prediction response."""

    hazard_year1: float
    hazard_year2: float
    hazard_year3: float
    n_snapshots: int = 0


def _load_normalizers(feature_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load snapshot + static normalizers. Returns (s_means, s_stds, st_means, st_stds)."""
    from bio_spread_reborn.data.dataset import (
        SNAPSHOT_FEATURE_COLS,
        STATIC_COLS,
        load_normalizers,
    )

    norm_path = feature_dir / "normalizers.npz"
    static_norm_path = feature_dir / "static_normalizers.npz"

    if norm_path.exists():
        s_means, s_stds = load_normalizers(norm_path)
    else:
        s_means, s_stds = np.zeros(len(SNAPSHOT_FEATURE_COLS)), np.ones(len(SNAPSHOT_FEATURE_COLS))

    if static_norm_path.exists():
        st_means, st_stds = load_normalizers(static_norm_path)
    else:
        st_means, st_stds = np.zeros(len(STATIC_COLS)), np.ones(len(STATIC_COLS))

    return s_means, s_stds, st_means, st_stds


def _features_to_tensor(
    req: PredictRequest, s_means: np.ndarray, s_stds: np.ndarray, st_means: np.ndarray, st_stds: np.ndarray, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Convert PredictRequest to model input tensors.

    Returns:
        static: (1, n_static) normalized static features
        seq: (1, L, n_snapshot) normalized snapshot features
        mask: (1, L) padding mask (all 1s)
        taxonomy: (1, 5) or None
    """
    # Snapshot features
    from bio_spread_reborn.data.dataset import SNAPSHOT_FEATURE_COLS, STATIC_COLS

    n_snap = len(req.snapshots)

    snap_arr = np.zeros((1, n_snap, len(SNAPSHOT_FEATURE_COLS)), dtype=np.float32)
    for i, snap in enumerate(req.snapshots):
        for j, col in enumerate(SNAPSHOT_FEATURE_COLS):
            snap_arr[0, i, j] = getattr(snap, col, 0.0) if hasattr(snap, col) else 0.0
    snap_arr = (snap_arr - s_means) / s_stds

    # Static features
    static_arr = np.zeros((1, len(STATIC_COLS)), dtype=np.float32)
    for j, col in enumerate(STATIC_COLS):
        static_arr[0, j] = getattr(req.static, col, 0.0)
    static_arr = (static_arr - st_means) / st_stds

    # Mask (all valid — single sequence)
    mask = torch.ones(1, n_snap, device=device)

    # Taxonomy
    taxonomy = None
    if req.taxonomy is not None:
        taxonomy = torch.tensor(
            [
                [
                    req.taxonomy.phylum_idx,
                    req.taxonomy.class_idx,
                    req.taxonomy.order_idx,
                    req.taxonomy.family_idx,
                    req.taxonomy.genus_idx,
                ]
            ],
            dtype=torch.long,
            device=device,
        )

    return (
        torch.from_numpy(static_arr).to(device),
        torch.from_numpy(snap_arr).to(device),
        mask,
        taxonomy,
    )


@app.on_event("startup")
def load_artifacts():
    global model, platt_scaler, cfg, feature_dims, _norm_cache

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    feature_dir = Path(FEATURE_DIR)
    s_means, s_stds, st_means, st_stds = _load_normalizers(feature_dir)
    _norm_cache = {
        "s_means": s_means,
        "s_stds": s_stds,
        "st_means": st_means,
        "st_stds": st_stds,
    }

    from bio_spread_reborn.data.dataset import SNAPSHOT_FEATURE_COLS, STATIC_COLS

    n_snapshot = len(SNAPSHOT_FEATURE_COLS)
    n_static = len(STATIC_COLS)

    # Taxonomy: load vocab if exists
    tax_vocab = None
    tax_vocab_path = feature_dir / "taxonomy_vocab.json"
    _TAXONOMY_RAW_COLS = ["TAXONOMY_phylum", "TAXONOMY_class", "TAXONOMY_order", "TAXONOMY_family", "genus"]
    if tax_vocab_path.exists():
        from bio_spread_reborn.data.snapshot import load_taxonomy_vocab

        tax_vocab = load_taxonomy_vocab(tax_vocab_path)
        logger.info("Taxonomy vocab loaded")

    # Build model using the factory
    from bio_spread_reborn.models import create_model
    from bio_spread_reborn.config.schema import ModelConfig as _ModelConfig
    model_cfg_raw = cfg.get("model", {})
    model_cfg = _ModelConfig(**model_cfg_raw)
    model = create_model(n_static, n_snapshot, model_cfg, taxonomy_vocab=tax_vocab)
    model.eval()

    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("Missing keys: %d (e.g., %s)", len(missing), list(missing)[:3])
    if unexpected:
        logger.warning("Unexpected keys: %d (e.g., %s)", len(unexpected), list(unexpected)[:3])
    model = model.to(DEVICE)

    # Platt scaler (optional)
    platt_path = PLATT_PATH or str(Path(MODEL_PATH).parent / "platt.pt")
    if os.path.exists(platt_path):
        platt_scaler = PlattScaler()
        platt_scaler.load_state_dict(torch.load(platt_path, map_location=DEVICE, weights_only=True))
        platt_scaler = platt_scaler.to(DEVICE)
        logger.info("Platt scaler loaded from %s", platt_path)

    feature_dims = {"n_static": n_static, "n_snapshot": n_snapshot}
    logger.info("SovereignX model loaded from %s (static=%d, snapshot=%d)", MODEL_PATH, n_static, n_snapshot)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Use cached normalizers (loaded at startup)
    s_means = _norm_cache.get("s_means", np.zeros(10))
    s_stds = _norm_cache.get("s_stds", np.ones(10))
    st_means = _norm_cache.get("st_means", np.zeros(10))
    st_stds = _norm_cache.get("st_stds", np.ones(10))
    static, seq, mask, taxonomy = _features_to_tensor(req, s_means, s_stds, st_means, st_stds, DEVICE)

    with torch.no_grad():
        out = model(static, seq, mask, taxonomy)

        # Apply Platt scaling if available
        logits = out.hazard_logits
        if platt_scaler is not None:
            logits = platt_scaler(logits)

        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    return PredictResponse(
        hazard_year1=float(probs[0]),
        hazard_year2=float(probs[1]),
        hazard_year3=float(probs[2]),
        n_snapshots=len(req.snapshots),
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "sovereign-x-pro",
        "feature_dims": feature_dims,
    }
