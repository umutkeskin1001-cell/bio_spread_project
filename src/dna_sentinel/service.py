from __future__ import annotations

from dna_sentinel.predict import predict_one
from dna_sentinel.train import load_checkpoint


class InferenceService:
    def __init__(self, checkpoint: str, device: str = "cpu") -> None:
        self.model = load_checkpoint(checkpoint, device=device)
        self.device = device

    def predict(self, sequence_id: str, dna: str) -> dict:
        return predict_one(self.model, sequence_id, dna, device=self.device).__dict__
