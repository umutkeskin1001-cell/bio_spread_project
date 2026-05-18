from __future__ import annotations

from pathlib import Path

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
            self.model = load_checkpoint(self.checkpoint_path, device=device)
            self.model_type = "neural"

    def predict(self, sequence_id: str, dna: str) -> dict:
        if self.model_type == "kmer":
            return self.model.predict_one(sequence_id, dna)
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

