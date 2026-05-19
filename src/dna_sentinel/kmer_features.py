"""Multi-scale k-mer and shift-invariant spectral feature extraction."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
        char_map = torch.full((256,), 4, dtype=torch.long)
        for idx, char in enumerate(["A", "C", "G", "T"]):
            char_map[ord(char)] = idx
            char_map[ord(char.lower())] = idx
        self.char_map = char_map
        self.multipliers = {
            k: 4 ** torch.arange(k - 1, -1, -1, dtype=torch.long)
            for k in range(ngram_min, ngram_max + 1)
        }

    def transform(self, windows: list[str], device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        n_windows = len(windows)
        if n_windows == 0:
            return (
                torch.zeros((0, self.n_features), dtype=torch.float32, device=device),
                torch.zeros((0, 512), dtype=torch.float32, device=device)
            )
        out_kmer = torch.zeros((n_windows, self.n_features), dtype=torch.float32, device=device)
        out_spec = torch.zeros((n_windows, 512), dtype=torch.float32, device=device)
        char_map = self.char_map.to(device)
        for idx, win in enumerate(windows):
            if not win:
                continue
            b = np.frombuffer(win.encode("ascii", errors="ignore"), dtype=np.uint8).copy()
            if len(b) < self.ngram_min:
                continue
            base4 = char_map[torch.from_numpy(b).long()].to(device)
            for k in range(self.ngram_min, min(self.ngram_max + 1, len(b) + 1)):
                kmers = base4.unfold(dimension=-1, size=k, step=1)
                valid = (kmers < 4).all(dim=-1)
                if not valid.any():
                    continue
                mult = self.multipliers[k].to(device)
                hashes = (kmers[valid] * mult).sum(dim=-1)
                hashed_indices = (hashes * 2654435761) % self.n_features
                out_kmer[idx] += torch.bincount(hashed_indices, minlength=self.n_features).float()
            one_hot = F.one_hot(base4, num_classes=5)[:, :4].float()
            fft_coefs = torch.fft.rfft(one_hot, dim=0)
            mags = torch.abs(fft_coefs)
            phases = torch.angle(fft_coefs)
            x_mags = mags.t().unsqueeze(0)
            x_phases = phases.t().unsqueeze(0)
            x_mags_i = F.interpolate(x_mags, size=64, mode="linear", align_corners=True).squeeze(0).t()
            x_phases_i = F.interpolate(x_phases, size=64, mode="linear", align_corners=True).squeeze(0).t()
            spec_feat = torch.cat([x_mags_i, x_phases_i], dim=0)
            out_spec[idx] = spec_feat.flatten()
        norms_kmer = torch.norm(out_kmer, p=2, dim=-1, keepdim=True).clamp_min(1e-8)
        norms_spec = torch.norm(out_spec, p=2, dim=-1, keepdim=True).clamp_min(1e-8)
        return out_kmer / norms_kmer, out_spec / norms_spec


class MultiScaleKmerExtractor:
    def __init__(self, config: MultiScaleKmerConfig | None = None):
        self.config = config or MultiScaleKmerConfig()
        self.extractor = PureTensorKmerExtractor(
            ngram_min=self.config.ngram_min,
            ngram_max=self.config.ngram_max,
            n_features=self.config.n_features,
        )

    def extract(self, dna: str, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        all_kmer, all_spec, all_mask, all_sid = [], [], [], []
        for idx, (ws, st, mw) in enumerate(zip(self.config.window_sizes, self.config.strides, self.config.max_windows)):
            windows = window_sequence(dna, ws, st, mw)
            kmer, spec = self.extractor.transform(windows, device=device)
            kmer, spec = kmer.cpu(), spec.cpu()
            if self.config.rc_consensus:
                rc_kmer, rc_spec = self.extractor.transform([revcomp(w) for w in windows], device=device)
                rc_kmer, rc_spec = rc_kmer.cpu(), rc_spec.cpu()
                kmer = 0.5 * (kmer + rc_kmer)
                spec = 0.5 * (spec + rc_spec)
            actual = kmer.shape[0]
            padded_kmer = torch.zeros(mw, self.config.n_features)
            padded_spec = torch.zeros(mw, 512)
            padded_kmer[:min(actual, mw)] = kmer[:mw]
            padded_spec[:min(actual, mw)] = spec[:mw]
            mask = torch.zeros(mw, dtype=torch.bool)
            mask[:min(actual, mw)] = True
            all_kmer.append(padded_kmer)
            all_spec.append(padded_spec)
            all_mask.append(mask)
            all_sid.append(torch.full((mw,), idx, dtype=torch.long))
        return torch.cat(all_kmer), torch.cat(all_spec), torch.cat(all_mask), torch.cat(all_sid)


def preprocess_all_features(records: list[LabeledSequence], config: MultiScaleKmerConfig, out_path: str | Path) -> None:
    extractor = MultiScaleKmerExtractor(config)
    feats, specs, masks, sids = [], [], [], []
    for rec in records:
        f, sp, m, s = extractor.extract(rec.dna)
        feats.append(f)
        specs.append(sp)
        masks.append(m)
        sids.append(s)
    torch.save({
        "features": torch.stack(feats),
        "spec_features": torch.stack(specs),
        "masks": torch.stack(masks),
        "scale_ids": torch.stack(sids)
    }, out_path)
