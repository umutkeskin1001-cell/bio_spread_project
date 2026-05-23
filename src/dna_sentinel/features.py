import os
from dataclasses import dataclass
from multiprocessing import get_all_start_methods, get_context
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dna_sentinel.utils import LabeledSequence

_FEATURE_WORKER: "MultiScaleKmerExtractor | None" = None


@dataclass(frozen=True)
class MultiScaleKmerConfig:
    window_sizes: tuple[int, ...] = (512, 2048, 8192)
    strides: tuple[int, ...] = (256, 1024, 4096)
    max_windows: tuple[int, ...] = (16, 8, 4)
    ngram_min: int = 4
    ngram_max: int = 6
    n_features: int = 5376  # Direct, collision-free vocabulary mapping (4-mer: 256, 5-mer: 1024, 6-mer: 4096 = 5376)
    rc_consensus: bool = True
    length_weighting: bool = False
    coverage_feature: bool = False


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


_BOUNDARIES_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def log_bin_spectral(mags: torch.Tensor, low_res: int = 64, n_bins: int = 64) -> torch.Tensor:
    mw, F_bins, C = mags.shape
    if F_bins <= low_res + n_bins:
        res = torch.zeros((mw, low_res + n_bins, C), device=mags.device)
        actual = min(F_bins, low_res + n_bins)
        res[:, :actual, :] = mags[:, :actual, :]
        return res

    low_part = mags[:, :low_res, :]

    key = (F_bins, low_res, n_bins)
    if key not in _BOUNDARIES_CACHE:
        boundaries = np.geomspace(low_res, F_bins, n_bins + 1).astype(int)
        for idx in range(1, len(boundaries)):
            if boundaries[idx] <= boundaries[idx - 1]:
                boundaries[idx] = boundaries[idx - 1] + 1
        _BOUNDARIES_CACHE[key] = boundaries
    else:
        boundaries = _BOUNDARIES_CACHE[key]

    high_parts = []
    for b in range(n_bins):
        start = boundaries[b]
        end = min(F_bins, boundaries[b + 1])
        if start < end:
            high_parts.append(mags[:, start:end, :].mean(dim=1))
        else:
            high_parts.append(torch.zeros((mw, C), device=mags.device))

    high_part = torch.stack(high_parts, dim=1)
    return torch.cat([low_part, high_part], dim=1)


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

    def _extract_strand(
        self,
        base4_seq: torch.Tensor,
        ws: int,
        st: int,
        mw: int,
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_bases = base4_seq.shape[0]

        if n_bases < ws:
            padded_seq = F.pad(base4_seq, (0, ws - n_bases), value=0)
            windows_tensor = padded_seq.unsqueeze(0)
            actual_lengths = torch.tensor([n_bases], dtype=torch.long, device=device)
            N = 1
        else:
            starts = list(range(0, n_bases - ws + 1, st))
            final_start = n_bases - ws
            if starts[-1] != final_start:
                starts.append(final_start)
            N = len(starts)
            starts_t = torch.tensor(starts, device=device, dtype=torch.long)
            indices = starts_t.unsqueeze(1) + torch.arange(ws, device=device)
            windows_tensor = base4_seq[indices]
            actual_lengths = torch.full((N,), ws, dtype=torch.long, device=device)

        out_kmer_all = torch.zeros((N, self.n_features), device=device)
        for k in range(self.config.ngram_min, self.config.ngram_max + 1):
            kmers = windows_tensor.unfold(dimension=-1, size=k, step=1)
            mult = self.multipliers[k].to(device)
            hashes = (kmers * mult).sum(dim=-1)

            # Robust fallback for custom unit test vocabulary sizes
            if self.n_features < 5376:
                hashed_indices = hashes % self.n_features
            else:
                if k == 4:
                    hashed_indices = hashes
                elif k == 5:
                    hashed_indices = 256 + hashes
                else:
                    hashed_indices = 1280 + hashes

            valid_kmers = torch.arange(ws - k + 1, device=device).unsqueeze(0) < (
                actual_lengths - k + 1
            ).clamp_min(0).unsqueeze(1)
            if not valid_kmers.any():
                continue

            offset = torch.arange(N, device=device).unsqueeze(1) * self.n_features
            flat_indices = (hashed_indices + offset)[valid_kmers]
            flat_counts = torch.bincount(flat_indices, minlength=N * self.n_features)
            out_kmer_all += flat_counts.view(N, self.n_features).float()

        norms_kmer = torch.norm(out_kmer_all, p=2, dim=-1, keepdim=True).clamp_min(1e-8)
        out_kmer_all = out_kmer_all / norms_kmer

        valid_bases = torch.arange(ws, device=device).unsqueeze(0) < actual_lengths.unsqueeze(1)
        one_hot = F.one_hot(windows_tensor.clamp(0, 3), num_classes=4).float()
        one_hot = one_hot * valid_bases.unsqueeze(-1)
        fft_coefs = torch.fft.rfft(one_hot, dim=1)
        mags = torch.abs(fft_coefs)
        phases = torch.angle(fft_coefs)

        mag_feat = log_bin_spectral(mags, low_res=32, n_bins=32).flatten(start_dim=1)
        phase_feat = log_bin_spectral(phases, low_res=32, n_bins=32).flatten(start_dim=1)
        out_spec_all = torch.cat([mag_feat, phase_feat], dim=1)

        norms_spec = torch.norm(out_spec_all, p=2, dim=-1, keepdim=True).clamp_min(1e-8)
        out_spec_all = out_spec_all / norms_spec
        coverage = (actual_lengths.float() / float(ws)).clamp(0.0, 1.0).unsqueeze(1)
        if self.config.length_weighting:
            out_kmer_all = out_kmer_all * coverage
            out_spec_all = out_spec_all * coverage
        if self.config.coverage_feature:
            out_spec_all[:, -1] = coverage.squeeze(1)

        if N > mw:
            out_kmer = torch.zeros((mw, self.n_features), device=device)
            out_spec = torch.zeros((mw, 512), device=device)
            mask = torch.ones(mw, dtype=torch.bool, device=device)

            bin_size = N / mw
            for i in range(mw):
                start_idx = int(round(i * bin_size))
                end_idx = max(start_idx + 1, int(round((i + 1) * bin_size)))
                out_kmer[i] = out_kmer_all[start_idx:end_idx].mean(dim=0)
                out_spec[i] = out_spec_all[start_idx:end_idx].mean(dim=0)
        else:
            out_kmer = torch.zeros((mw, self.n_features), device=device)
            out_spec = torch.zeros((mw, 512), device=device)
            mask = torch.zeros(mw, dtype=torch.bool, device=device)
            out_kmer[:N] = out_kmer_all
            out_spec[:N] = out_spec_all
            mask[:N] = True

        out_kmer = out_kmer * mask.unsqueeze(1).float()
        out_spec = out_spec * mask.unsqueeze(1).float()

        return out_kmer, out_spec, mask

    def extract(self, dna: str, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b = np.frombuffer(dna.encode("ascii", errors="ignore"), dtype=np.uint8).copy()
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


def _init_feature_worker(config: MultiScaleKmerConfig) -> None:
    global _FEATURE_WORKER
    _FEATURE_WORKER = MultiScaleKmerExtractor(config)


def _extract_record_features(dna: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if _FEATURE_WORKER is None:
        raise RuntimeError("Feature worker was not initialized")
    return _FEATURE_WORKER.extract(dna)


def preprocess_all_features(
    records: list[LabeledSequence],
    config: MultiScaleKmerConfig,
    out_path: str | Path,
    num_workers: int | None = None,
    parallel_threshold: int = 32,
) -> None:
    extractor = MultiScaleKmerExtractor(config)
    if num_workers is None:
        num_workers = os.cpu_count() or 1
    use_parallel = num_workers > 1 and len(records) >= parallel_threshold

    if use_parallel:
        start_method = "fork" if "fork" in get_all_start_methods() else "spawn"
        ctx = get_context(start_method)
        with ctx.Pool(processes=num_workers, initializer=_init_feature_worker, initargs=(config,)) as pool:
            extracted = pool.map(_extract_record_features, [rec.dna for rec in records])
    else:
        extracted = [extractor.extract(rec.dna) for rec in records]

    feats, specs, masks, sids = zip(*extracted, strict=True) if extracted else ([], [], [], [])
    torch.save({
        "features": torch.stack(feats),
        "spec_features": torch.stack(specs),
        "masks": torch.stack(masks),
        "scale_ids": torch.stack(sids)
    }, out_path)
