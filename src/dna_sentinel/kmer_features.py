"""Multi-scale k-mer feature extraction."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from sklearn.feature_extraction.text import HashingVectorizer
from dna_sentinel.dataset import LabeledSequence
from dna_sentinel.fasta import revcomp
from dna_sentinel.tokenizer import window_sequence

@dataclass(frozen=True)
class MultiScaleKmerConfig:
    window_sizes: tuple[int, ...] = (512, 2048, 8192)
    strides: tuple[int, ...] = (256, 1024, 4096)
    max_windows: tuple[int, ...] = (16, 8, 4)
    ngram_min: int = 4
    ngram_max: int = 6
    n_features: int = 4096
    rc_consensus: bool = True

class MultiScaleKmerExtractor:
    def __init__(self, config: MultiScaleKmerConfig | None = None):
        self.config = config or MultiScaleKmerConfig()
        self.vectorizers = [
            HashingVectorizer(
                analyzer="char",
                ngram_range=(self.config.ngram_min, self.config.ngram_max),
                n_features=self.config.n_features,
                alternate_sign=False,
                norm="l2",
                lowercase=False,
            )
            for _ in range(len(self.config.window_sizes))
        ]

    def extract(self, dna: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        all_feat, all_mask, all_sid = [], [], []
        for idx, (ws, st, mw) in enumerate(zip(self.config.window_sizes, self.config.strides, self.config.max_windows)):
            windows = window_sequence(dna, ws, st, mw)
            feat = self.vectorizers[idx].transform(windows).toarray().astype(np.float32)
            if self.config.rc_consensus:
                rc_feat = self.vectorizers[idx].transform([revcomp(w) for w in windows]).toarray().astype(np.float32)
                feat = 0.5 * (feat + rc_feat)
            actual = feat.shape[0]
            padded = torch.zeros(mw, self.config.n_features)
            padded[:min(actual, mw)] = torch.from_numpy(feat[:mw])
            mask = torch.zeros(mw, dtype=torch.bool)
            mask[:min(actual, mw)] = True
            all_feat.append(padded)
            all_mask.append(mask)
            all_sid.append(torch.full((mw,), idx, dtype=torch.long))
        return torch.cat(all_feat), torch.cat(all_mask), torch.cat(all_sid)

def preprocess_all_features(records: list[LabeledSequence], config: MultiScaleKmerConfig, out_path: str | Path) -> None:
    extractor = MultiScaleKmerExtractor(config)
    feats, masks, sids = [], [], []
    for rec in records:
        f, m, s = extractor.extract(rec.dna)
        feats.append(f)
        masks.append(m)
        sids.append(s)
    torch.save({"features": torch.stack(feats), "masks": torch.stack(masks), "scale_ids": torch.stack(sids)}, out_path)
