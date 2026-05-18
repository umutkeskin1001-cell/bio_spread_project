"""Sequence augmentation: RC mirroring, window dropout, and circular permutation."""
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

    def __call__(self, features: torch.Tensor | list[torch.Tensor], mask: torch.Tensor, training: bool = True):
        if not training:
            return features, mask
        B, W = mask.shape[:2]
        keep = (torch.rand((B, W), device=mask.device) >= self.drop_rate).float()
        keep[:, 0] = 1.0
        keep_u = keep.unsqueeze(-1)
        if isinstance(features, list):
            return [feat * keep_u for feat in features], mask & keep.bool()
        return features * keep_u, mask & keep.bool()
