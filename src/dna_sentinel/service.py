from __future__ import annotations

import math
from pathlib import Path

import torch

from dna_sentinel.fasta import canonical_dna
from dna_sentinel.predict import predict_one as predict_neural
from dna_sentinel.train import load_checkpoint


class InferenceService:
    def __init__(self, checkpoint: str, device: str = "cpu") -> None:
        self.device = device
        self.checkpoint_path = Path(checkpoint)
        if self.checkpoint_path.suffix == ".joblib":
            from dna_sentinel.kmer import KmerSentinel
            self.model = KmerSentinel.load(self.checkpoint_path)
            self.model_type = "kmer"
        else:
            state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            if "config" in state and "state_dict" in state:
                from dna_sentinel.kmer_transformer import KmerTransformer
                self.model = KmerTransformer.load(self.checkpoint_path, device=device)
                self.model.to(device)
                self.model.eval()
                self.model_type = "kmer_transformer"
            else:
                self.model = load_checkpoint(self.checkpoint_path, device=device)
                self.model_type = "neural"

    @torch.inference_mode()
    def predict(self, sequence_id: str, dna: str) -> dict:
        dna = canonical_dna(dna)
        if self.model_type == "kmer":
            return self.model.predict_one(sequence_id, dna)
        elif self.model_type == "kmer_transformer":
            from dna_sentinel.kmer_features import MultiScaleKmerConfig, MultiScaleKmerExtractor
            from dna_sentinel.tokenizer import window_sequence
            extractor = MultiScaleKmerExtractor(MultiScaleKmerConfig(n_features=self.model.config.n_kmer_features))
            feat, spec, mask, sid = extractor.extract(dna)
            feat = feat.unsqueeze(0).to(self.device)
            spec = spec.unsqueeze(0).to(self.device)
            mask = mask.unsqueeze(0).to(self.device)
            sid = sid.unsqueeze(0).to(self.device)
            out = self.model(feat, spec, mask, sid)
            mobility = torch.softmax(out["mobility_logits"], dim=-1).squeeze(0).cpu().tolist()
            amr = float(torch.sigmoid(out["amr_logits"]).item())
            expansion = float(torch.sigmoid(out["expansion_logits"]).item())
            mobile = max(mobility[1], mobility[2])
            risk = float(((mobile**2 + amr**2 + expansion**2) / 3.0)**0.5)

            weights = out["evidence_weights"].squeeze(0).cpu()
            active_mask = mask.squeeze(0).cpu().bool()
            all_windows_info = []
            for ws, st, mw in zip(extractor.config.window_sizes, extractor.config.strides, extractor.config.max_windows):
                windows = window_sequence(dna, ws, st, mw)
                for i in range(mw):
                    if i < len(windows):
                        w = windows[i]
                        start = i * st
                        all_windows_info.append({"start": float(start), "end": float(start + len(w))})
                    else:
                        all_windows_info.append({"start": 0.0, "end": 0.0})

            sorted_indices = torch.argsort(weights, descending=True)
            top_windows = []
            for idx in sorted_indices.tolist():
                if len(top_windows) >= 5:
                    break
                if active_mask[idx]:
                    info = all_windows_info[idx]
                    top_windows.append({"start": info["start"], "end": info["end"], "weight": float(weights[idx])})

            return {
                "sequence_id": sequence_id,
                "mobility_probs": mobility,
                "amr_probability": amr,
                "expansion_probability": expansion,
                "risk_score": risk,
                "top_windows": top_windows,
            }
        else:
            pred = predict_neural(self.model, sequence_id, dna, device=self.device)
            return {
                "sequence_id": pred.sequence_id,
                "mobility_probs": pred.mobility_probs,
                "amr_probability": pred.amr_probability,
                "expansion_probability": pred.expansion_probability,
                "risk_score": pred.risk_score,
                "top_windows": pred.top_windows,
            }
