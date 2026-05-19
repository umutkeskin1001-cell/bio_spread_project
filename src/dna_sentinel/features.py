"""Multi-Scale DNA feature extraction using vectorized 2D PyTorch operations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dna_sentinel.utils import LabeledSequence


def window_sequence(seq: str, window: int, stride: int, max_windows: int) -> list[str]:
    if not seq:
        return [""]
    if len(seq) <= window:
        return [seq]
    starts = list(range(0, max(1, len(seq) - window + 1), stride))
    if starts[-1] != len(seq) - window:
        starts.append(len(seq) - window)
    if len(starts) > max_windows:
        if max_windows == 1:
            starts = [starts[len(starts) // 2]]
        else:
            step = (len(starts) - 1) / (max_windows - 1)
            starts = [starts[round(i * step)] for i in range(max_windows)]
    return [seq[s : s + window] for s in starts]


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
        self.n_features = self.config.n_features
        self.char_map = torch.zeros(256, dtype=torch.long)
        for idx, char in enumerate(["A", "C", "G", "T"]):
            self.char_map[ord(char)] = idx
            self.char_map[ord(char.lower())] = idx

        self.multipliers = {
            k: 4 ** torch.arange(k - 1, -1, -1, dtype=torch.long)
            for k in range(self.config.ngram_min, self.config.ngram_max + 1)
        }

    def _extract_strand(self, base4_seq: torch.Tensor, ws: int, st: int, mw: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_bases = base4_seq.shape[0]
        if n_bases <= ws:
            starts = [0]
        else:
            starts = list(range(0, n_bases - ws + 1, st))
            if starts[-1] != n_bases - ws:
                starts.append(n_bases - ws)
            if len(starts) > mw:
                step = (len(starts) - 1) / (mw - 1)
                starts = [starts[round(i * step)] for i in range(mw)]

        windows_tensor = torch.zeros((mw, ws), dtype=torch.long, device=device)
        mask = torch.zeros(mw, dtype=torch.bool, device=device)
        for i, s in enumerate(starts):
            win = base4_seq[s : s + ws]
            actual_len = min(len(win), ws)
            windows_tensor[i, :actual_len] = win[:actual_len]
            mask[i] = True

        out_kmer = torch.zeros((mw, self.n_features), device=device)
        for k in range(self.config.ngram_min, self.config.ngram_max + 1):
            kmers = windows_tensor.unfold(dimension=-1, size=k, step=1)
            mult = self.multipliers[k].to(device)
            hashes = (kmers * mult).sum(dim=-1)
            hashed_indices = (hashes * 2654435761) % self.n_features

            offset = torch.arange(mw, device=device).unsqueeze(1) * self.n_features
            flat_indices = hashed_indices + offset
            flat_counts = torch.bincount(flat_indices.flatten(), minlength=mw * self.n_features)
            out_kmer += flat_counts.view(mw, self.n_features).float()

        norms_kmer = torch.norm(out_kmer, p=2, dim=-1, keepdim=True).clamp_min(1e-8)
        out_kmer = out_kmer / norms_kmer

        one_hot = F.one_hot(windows_tensor.clamp(0, 3), num_classes=4).float()
        fft_coefs = torch.fft.rfft(one_hot, dim=1)
        mags = torch.abs(fft_coefs)

        K = 128
        spec_feat = mags[:, :K, :]
        out_spec = spec_feat.flatten(start_dim=1)

        norms_spec = torch.norm(out_spec, p=2, dim=-1, keepdim=True).clamp_min(1e-8)
        out_spec = out_spec / norms_spec

        out_kmer = out_kmer * mask.unsqueeze(1).float()
        out_spec = out_spec * mask.unsqueeze(1).float()

        return out_kmer, out_spec, mask

    def extract(self, dna: str, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b = np.frombuffer(dna.encode("ascii", errors="ignore"), dtype=np.uint8)
        base4_f = self.char_map[torch.from_numpy(b).long()].to(device)

        all_kmer, all_spec, all_mask, all_sid = [], [], [], []
        for idx, (ws, st, mw) in enumerate(zip(self.config.window_sizes, self.config.strides, self.config.max_windows)):
            kmer_f, spec_f, mask = self._extract_strand(base4_f, ws, st, mw, device)

            if self.config.rc_consensus:
                base4_rc = 3 - torch.flip(base4_f, dims=[0])
                kmer_rc, spec_rc, _ = self._extract_strand(base4_rc, ws, st, mw, device)
                kmer = 0.5 * (kmer_f + kmer_rc)
                spec = 0.5 * (spec_f + spec_rc)
            else:
                kmer, spec = kmer_f, spec_f

            all_kmer.append(kmer.cpu())
            all_spec.append(spec.cpu())
            all_mask.append(mask.cpu())
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
