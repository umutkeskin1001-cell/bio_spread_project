"""Sequence augmentation: RC mirroring and window dropout."""
import torch

from dna_sentinel.dataset import LabeledSequence
from dna_sentinel.fasta import revcomp


def rc_augment(records: list[LabeledSequence]) -> list[LabeledSequence]:
    augmented = []
    for rec in records:
        augmented.append(rec)
        augmented.append(LabeledSequence(
            sequence_id=f"{rec.sequence_id}_rc",
            dna=revcomp(rec.dna),
            mobility=rec.mobility,
            amr=rec.amr,
            expansion=rec.expansion,
        ))
    return augmented

class WindowDropout:
    def __init__(self, drop_rate: float = 0.25):
        self.drop_rate = drop_rate

    def __call__(self, features: torch.Tensor, mask: torch.Tensor, training: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        if not training:
            return features, mask
        B, W = features.shape[:2]
        keep = torch.bernoulli(torch.full((B, W), 1 - self.drop_rate, device=features.device))
        keep[:, 0] = 1.0
        return features * keep.unsqueeze(-1), mask & keep.bool()
