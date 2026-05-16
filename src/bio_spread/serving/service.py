from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

from bio_spread.config.schema import ModelConfig
from bio_spread.constants import (
    ALL_SNAPSHOT_COLS,
    HEAVY_TAILED_FEATURES,
    SNAPSHOT_FEATURE_COLS,
    SNAPSHOT_NAN_COLS,
    STATIC_COLS,
)
from bio_spread.data.dataset import load_normalizers
from bio_spread.data.snapshot import load_taxonomy_vocab
from bio_spread.models import create_model
from bio_spread.models.components import PlattScaler

logger = logging.getLogger(__name__)


class InferenceService:
    def __init__(
        self,
        model_path: str,
        config_path: str,
        feature_dir: str,
        platt_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        feature_dir = Path(feature_dir)

        with open(config_path) as f:
            cfg_raw = yaml.safe_load(f)
        self.model_cfg = ModelConfig(**cfg_raw.get("model", {}))

        n_expected = len(ALL_SNAPSHOT_COLS)
        try:
            self.s_means, self.s_stds = load_normalizers(feature_dir / "normalizers.npz")
        except FileNotFoundError:
            logger.warning(
                "Snapshot normalizers not found at %s — using identity (zeros/ones)",
                feature_dir / "normalizers.npz",
            )
            self.s_means = np.zeros(n_expected)
            self.s_stds = np.ones(n_expected)

        try:
            self.st_means, self.st_stds = load_normalizers(feature_dir / "static_normalizers.npz")
        except FileNotFoundError:
            logger.warning(
                "Static normalizers not found at %s — using identity (zeros/ones)",
                feature_dir / "static_normalizers.npz",
            )
            self.st_means = np.zeros(len(STATIC_COLS))
            self.st_stds = np.ones(len(STATIC_COLS))

        if len(self.s_means) < n_expected:
            n_pad = n_expected - len(self.s_means)
            self.s_means = np.concatenate([self.s_means, np.zeros(n_pad)])
            self.s_stds = np.concatenate([self.s_stds, np.ones(n_pad)])

        n_snapshot = len(ALL_SNAPSHOT_COLS)
        n_static = len(STATIC_COLS)

        tax_vocab_path = feature_dir / "taxonomy_vocab.json"
        taxonomy_vocab = None
        if tax_vocab_path.exists():
            taxonomy_vocab = load_taxonomy_vocab(tax_vocab_path)

        self.model = create_model(
            n_static, n_snapshot, self.model_cfg,
            taxonomy_vocab=taxonomy_vocab,
        )
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        missing_keys, unexpected_keys = self.model.load_state_dict(state, strict=False)
        if missing_keys:
            logger.warning("Missing keys loading checkpoint: %s", missing_keys[:10])
        if unexpected_keys:
            logger.warning("Unexpected keys loading checkpoint: %s", unexpected_keys[:10])
        self.model.eval()
        self.model.to(self.device)

        self.platt_scalers: Optional[List[PlattScaler]] = None
        platt_path_resolved = platt_path or str(Path(model_path).parent / "platt.pt")
        if Path(platt_path_resolved).exists():
            platt_state = torch.load(platt_path_resolved, map_location=self.device, weights_only=True)
            scalers = [PlattScaler().to(self.device) for _ in range(3)]
            for h in range(3):
                key = f"scaler_h{h}"
                if key in platt_state:
                    scalers[h].load_state_dict(platt_state[key])
            self.platt_scalers = scalers

        self.cold_platt_scalers: Optional[List[PlattScaler]] = None
        platt_cold_path = Path(model_path).parent / "platt_cold.pt"
        if platt_cold_path.exists():
            platt_state = torch.load(str(platt_cold_path), map_location=self.device, weights_only=True)
            scalers = [PlattScaler().to(self.device) for _ in range(3)]
            for h in range(3):
                key = f"scaler_h{h}"
                if key in platt_state:
                    scalers[h].load_state_dict(platt_state[key])
            self.cold_platt_scalers = scalers

    def _features_to_tensor(
        self,
        snapshots: List[Dict[str, float]],
        static: Dict[str, float],
        taxonomy: Optional[Dict[str, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        n_snap = len(snapshots)
        n_base = len(SNAPSHOT_FEATURE_COLS)
        n_nan = len(SNAPSHOT_NAN_COLS)
        n_total = n_base + n_nan

        snap_arr = np.zeros((1, n_snap, n_total), dtype=np.float32)
        for j, col in enumerate(SNAPSHOT_FEATURE_COLS):
            for t, snap in enumerate(snapshots):
                v = snap.get(col, 0.0)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    v = float(v)
                    if col in HEAVY_TAILED_FEATURES:
                        v = float(np.log1p(max(v, 0)))
                    snap_arr[0, t, j] = v
                    snap_arr[0, t, n_base + j] = 0.0
                else:
                    snap_arr[0, t, j] = 0.0
                    snap_arr[0, t, n_base + j] = 1.0

        snap_arr[0] = ((snap_arr[0] - self.s_means) / self.s_stds).astype(np.float32)

        static_arr = np.zeros((1, len(STATIC_COLS)), dtype=np.float32)
        for j, col in enumerate(STATIC_COLS):
            v = static.get(col, 0.0)
            static_arr[0, j] = float(v) if v is not None else 0.0
        static_arr = ((static_arr - self.st_means) / self.st_stds).astype(np.float32)

        taxonomy_arr = None
        if taxonomy is not None:
            taxonomy_arr = np.ones((1, 5), dtype=np.int64)
            tax_keys = ["phylum_idx", "class_idx", "order_idx", "family_idx", "genus_idx"]
            for j, key in enumerate(tax_keys):
                v = taxonomy.get(key, 0)
                taxonomy_arr[0, j] = int(v) if v is not None else 0
        elif self.model.use_taxonomy:
            taxonomy_arr = np.ones((1, 5), dtype=np.int64)

        mask = np.ones((1, n_snap), dtype=np.float32)

        return {
            "static": torch.from_numpy(static_arr).to(self.device),
            "seq": torch.from_numpy(snap_arr).to(self.device),
            "mask": torch.from_numpy(mask).to(self.device),
            "taxonomy": torch.from_numpy(taxonomy_arr).long().to(self.device) if taxonomy_arr is not None else None,
        }

    def _predict(
        self, snapshots, static, taxonomy, logits_key: str = "hazard_logits", use_cold: bool = False
    ) -> Dict[str, float]:
        if not snapshots:
            raise ValueError("Need at least 1 snapshot for prediction")
        missing_snap = [c for c in SNAPSHOT_FEATURE_COLS if c not in snapshots[0]]
        if missing_snap:
            raise ValueError(f"Missing snapshot feature keys: {missing_snap}")
        missing_static = [c for c in STATIC_COLS if c not in static]
        if missing_static:
            raise ValueError(f"Missing static feature keys: {missing_static}")

        tensors = self._features_to_tensor(snapshots, static, taxonomy)
        with torch.no_grad():
            out = self.model(tensors["static"], tensors["seq"], tensors["mask"], tensors["taxonomy"])
        logits = getattr(out, logits_key)
        scalers = self.cold_platt_scalers if use_cold else self.platt_scalers

        if scalers is not None:
            probs = torch.zeros(3, device=self.device)
            for h in range(3):
                probs[h] = torch.sigmoid(scalers[h](logits[:, h])).squeeze()
        else:
            probs = torch.sigmoid(logits).squeeze()
        p = probs.cpu().numpy().flatten()
        return {"hazard_year1": float(p[0]), "hazard_year2": float(p[1]), "hazard_year3": float(p[2]), "n_snapshots": len(snapshots)}

    def predict(self, snapshots, static, taxonomy=None):
        return self._predict(snapshots, static, taxonomy, "hazard_logits")

    def predict_with_cold_scalers(self, snapshots, static, taxonomy=None):
        return self._predict(snapshots, static, taxonomy, "cold_logits", use_cold=True)
