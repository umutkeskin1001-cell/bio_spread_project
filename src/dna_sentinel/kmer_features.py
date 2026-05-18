"""Multi-scale k-mer feature extraction."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

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

class PureTensorKmerExtractor:
    def __init__(self, ngram_min: int = 4, ngram_max: int = 6, n_features: int = 4096):
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.n_features = n_features
        char_map = torch.zeros(256, dtype=torch.long)
        for idx, char in enumerate(["A", "C", "G", "T"]):
            char_map[ord(char)] = idx
            char_map[ord(char.lower())] = idx
        self.char_map = char_map
        self.multipliers = {
            k: 4 ** torch.arange(k - 1, -1, -1, dtype=torch.long)
            for k in range(ngram_min, ngram_max + 1)
        }

    def transform(self, windows: list[str], device: str = "cpu") -> torch.Tensor:
        n_windows = len(windows)
        if n_windows == 0:
            return torch.zeros((0, self.n_features), dtype=torch.float32, device=device)
        out = torch.zeros((n_windows, self.n_features), dtype=torch.float32, device=device)
        char_map = self.char_map.to(device)
        for idx, win in enumerate(windows):
            if not win:
                continue
            b = np.frombuffer(win.encode("ascii", errors="ignore"), dtype=np.uint8).copy()
            if len(b) < self.ngram_min:
                continue
            base4 = char_map[torch.from_numpy(b).long()]
            for k in range(self.ngram_min, min(self.ngram_max + 1, len(b) + 1)):
                kmers = base4.unfold(dimension=-1, size=k, step=1)
                mult = self.multipliers[k].to(device)
                hashes = (kmers * mult).sum(dim=-1)
                hashed_indices = (hashes * 2654435761) % self.n_features
                freqs = torch.bincount(hashed_indices, minlength=self.n_features).float()
                out[idx] += freqs
        norms = torch.norm(out, p=2, dim=-1, keepdim=True).clamp_min(1e-8)
        return out / norms

class MultiScaleKmerExtractor:
    def __init__(self, config: MultiScaleKmerConfig | None = None):
        self.config = config or MultiScaleKmerConfig()
        self.extractor = PureTensorKmerExtractor(
            ngram_min=self.config.ngram_min,
            ngram_max=self.config.ngram_max,
            n_features=self.config.n_features,
        )

    def extract(self, dna: str, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        all_feat, all_mask, all_sid = [], [], []
        for idx, (ws, st, mw) in enumerate(zip(self.config.window_sizes, self.config.strides, self.config.max_windows)):
            windows = window_sequence(dna, ws, st, mw)
            feat = self.extractor.transform(windows, device=device).cpu()
            if self.config.rc_consensus:
                rc_feat = self.extractor.transform([revcomp(w) for w in windows], device=device).cpu()
                feat = 0.5 * (feat + rc_feat)
            actual = feat.shape[0]
            padded = torch.zeros(mw, self.config.n_features)
            padded[:min(actual, mw)] = feat[:mw]
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
