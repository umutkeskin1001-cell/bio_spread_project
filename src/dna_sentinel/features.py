from __future__ import annotations

import functools
import logging
import os
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dna_sentinel.utils import LabeledSequence, ValidationError, circular_shift, load_jsonl, revcomp

_logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=None)
def _canonical_vocab(k: int) -> int:
    return ((4**k) + (4 ** (k // 2) if k % 2 == 0 else 0)) // 2


@dataclass(frozen=True)
class CanonicalKmerConfig:
    window_sizes: tuple[int, ...] = (512, 2048, 8192)
    strides: tuple[int, ...] = (256, 1024, 4096)
    max_windows: tuple[int, ...] = (16, 8, 4)
    ngram_min: int = 4
    ngram_max: int = 6
    rc_consensus: bool = False
    n_features: int | None = None
    n_structural_features: int = 19


from functools import lru_cache


@lru_cache(maxsize=8)
def _resolve_max_windows(max_windows: int | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if not isinstance(max_windows, int):
        return tuple(max_windows)
    base = (16, 8, 4)
    total = max_windows
    out = [max(1, round(total * b / sum(base))) for b in base]
    while sum(out) != total:
        i = max(range(len(base)), key=lambda j: base[j] if sum(out) < total else out[j])
        out[i] += 1 if sum(out) < total else -1
    return tuple(out)


@functools.lru_cache(maxsize=None)
def _canonical_map(k: int) -> torch.Tensor:
    n = 4**k
    cmap = torch.zeros(n, dtype=torch.long)
    assigned, idx = {}, 0
    for i in range(n):
        if i in assigned:
            cmap[i] = assigned[i]
            continue
        val, rc = i, 0
        for _ in range(k):
            rc = rc * 4 + (3 - val % 4)
            val //= 4
        assigned[i] = assigned[rc] = idx
        cmap[i] = idx
        if i != rc:
            cmap[rc] = idx
        idx += 1
    return cmap


def _vocab_offsets(ngram_min: int, ngram_max: int) -> dict[int, int]:
    off, offsets = 0, {}
    for k in range(ngram_min, ngram_max + 1):
        offsets[k] = off
        off += _canonical_vocab(k)
    return offsets


class CanonicalKmerExtractor:
    def __init__(self, config: CanonicalKmerConfig | None = None):
        self.config = config or CanonicalKmerConfig()
        self.char_map = torch.full((256,), -1, dtype=torch.long)
        for idx, ch in enumerate("ACGT"):
            self.char_map[ord(ch)] = self.char_map[ord(ch.lower())] = idx
        self._km_range = range(self.config.ngram_min, self.config.ngram_max + 1)
        self._mult = {k: 4 ** torch.arange(k - 1, -1, -1, dtype=torch.long) for k in self._km_range}
        self._cmap = {k: _canonical_map(k) for k in self._km_range}
        self._offsets = _vocab_offsets(self.config.ngram_min, self.config.ngram_max)
        self.n_features = sum(_canonical_vocab(k) for k in self._km_range)

    def _struct(self, w: torch.Tensor, lengths: torch.Tensor, ws: int) -> torch.Tensor:
        dev = w.device
        v = (torch.arange(ws, device=dev).unsqueeze(0) < lengths.unsqueeze(1)) & (w >= 0)
        wc = w.clamp_min(0)
        g, c, a, t = (wc == 2).float(), (wc == 1).float(), (wc == 0).float(), (wc == 3).float()
        vf = v.float()
        gc_sum = ((g + c) * vf).sum(dim=1)
        gc = gc_sum / lengths.float().clamp_min(1)
        gs = ((g - c) * vf).sum(dim=1) / gc_sum.clamp_min(1)
        at_s = ((a - t) * vf).sum(dim=1) / ((a + t) * vf).sum(dim=1).clamp_min(1)
        pv = v[:, :-1] & v[:, 1:]
        di = (F.one_hot((wc[:, :-1] * 4 + wc[:, 1:]).long().clamp_max(15), 16).float() * pv.unsqueeze(-1)).sum(dim=1)
        di = di / di.sum(dim=-1, keepdim=True).clamp_min(1)
        return torch.cat([gc.unsqueeze(-1), gs.unsqueeze(-1), at_s.unsqueeze(-1), di], dim=-1)

    def _extract(self, seq: torch.Tensor, ws: int, st: int, mw: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dev, n = seq.device, seq.shape[0]
        seq = seq.clamp_min(0)
        if n < ws:
            w = F.pad(seq, (0, ws - n)).unsqueeze(0)
            lengths = torch.tensor([n], dtype=torch.long, device=dev)
            N = 1
        else:
            starts = list(range(0, n - ws + 1, st))
            if starts[-1] != n - ws:
                starts.append(n - ws)
            N = len(starts)
            idx = torch.tensor(starts, device=dev, dtype=torch.long)[:, None] + torch.arange(ws, device=dev)
            w = torch.cat([seq, seq[: ws - 1]])[idx]
            lengths = torch.full((N,), ws, dtype=torch.long, device=dev)
        struct = self._struct(w, lengths, ws)
        nf = self.n_features
        raw = torch.zeros(N, nf, device=dev)
        mult_d = {k: v.to(dev) for k, v in self._mult.items()}
        cmap_d = {k: v.to(dev) for k, v in self._cmap.items()}
        for k in self._km_range:
            m = ws - k + 1
            kmers = w.unfold(-1, k, 1)
            hashes = (kmers * mult_d[k]).sum(dim=-1)
            canon = cmap_d[k][hashes] + self._offsets[k]
            valid = torch.arange(m, device=dev).unsqueeze(0) < (lengths - k + 1).clamp_min(0).unsqueeze(1)
            if not valid.any():
                continue
            raw += (
                torch.bincount((canon + torch.arange(N, device=dev).unsqueeze(1) * nf)[valid], minlength=N * nf)
                .view(N, nf)
                .float()
            )
        if N > mw:
            bs = N / mw
            bnd = (torch.arange(mw + 1, device=dev, dtype=torch.float32) * bs).round().long().clamp_max(N)
            cum, cum_s = raw.cumsum(dim=0), struct.cumsum(dim=0)
            cnt = (bnd[1:] - bnd[:-1]).clamp_min(1)
            out = (cum[bnd[1:] - 1] - F.pad(cum[:-1], (0, 0, 1, 0))[bnd[:-1]]) / cnt[:, None]
            out_s = (cum_s[bnd[1:] - 1] - F.pad(cum_s[:-1], (0, 0, 1, 0))[bnd[:-1]]) / cnt[:, None]
            out = F.normalize(out + 1e-6, p=2, dim=-1)
            mask = torch.ones(mw, dtype=torch.bool, device=dev)
        else:
            out = torch.zeros(mw, nf, device=dev)
            out_s = torch.zeros(mw, self.config.n_structural_features, device=dev)
            out[:N] = F.normalize(raw + 1e-6, p=2, dim=-1)
            out_s[:N] = struct
            mask = torch.zeros(mw, dtype=torch.bool, device=dev)
            mask[:N] = True
        return out * mask.unsqueeze(1), out_s * mask.unsqueeze(1), mask

    def extract(self, dna: str, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not dna:
            raise ValidationError("empty DNA string in feature extraction")
        b = np.frombuffer(dna.encode("ascii", errors="replace"), dtype=np.uint8).copy()
        seq = self.char_map[torch.from_numpy(b).long()].to(device)
        n_count = (seq < 0).sum().item()
        if n_count:
            _logger.warning("extract: %d non-ACGT base(s) found (mapping to A, recording N ratio)", n_count)
            n_frac = n_count / max(1, seq.numel())
            seq = seq.clamp_min(0)
        else:
            n_frac = 0.0
        rc_seq = (3 - torch.flip(seq, dims=[0])).clamp(0, 3) if self.config.rc_consensus else None
        feat_l, struct_l, mask_l, sid_l = [], [], [], []
        for idx, (ws, st, mw) in enumerate(zip(self.config.window_sizes, self.config.strides, self.config.max_windows)):
            f, s, m = self._extract(seq, ws, st, mw)
            if rc_seq is not None:
                fr, sr, _ = self._extract(rc_seq, ws, st, mw)
                f, s = 0.5 * (f + fr), 0.5 * (s + sr)
            feat_l.append(f.cpu())
            struct_l.append(s.cpu())
            mask_l.append(m.cpu())
            sid_l.append(torch.full((mw,), idx, dtype=torch.long))
        return torch.cat(feat_l), torch.cat(struct_l), torch.cat(mask_l), torch.cat(sid_l)


_WORKER: CanonicalKmerExtractor | None = None


def _init_worker(cfg: CanonicalKmerConfig):
    global _WORKER
    _WORKER = CanonicalKmerExtractor(cfg)


def _run(dna: str):
    if _WORKER is None:
        raise RuntimeError("worker not initialized")
    return _WORKER.extract(dna)


def _consistency_transform(record: LabeledSequence, index: int) -> str:
    if index % 2:
        return revcomp(record.dna)
    return circular_shift(record.dna, max(1, len(record.dna) // 2))


def preprocess_all_features(
    records: list[LabeledSequence],
    config: CanonicalKmerConfig,
    out_path: str | Path,
    num_workers: int | None = None,
    parallel_threshold: int = 32,
) -> None:
    if not records:
        _logger.warning("preprocess_all_features: empty records, skipping")
        return
    if num_workers is None:
        num_workers = os.cpu_count() or 1
    use = num_workers > 1 and len(records) >= parallel_threshold
    if use:
        ctx = get_context("spawn")
        with ctx.Pool(processes=num_workers, initializer=_init_worker, initargs=(config,)) as p:
            extracted = p.map(_run, [r.dna for r in records])
    else:
        ex = CanonicalKmerExtractor(config)
        extracted = [ex.extract(r.dna) for r in records]
    feats, structs, masks, sids = zip(*extracted)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": torch.stack(feats),
            "struct_features": torch.stack(structs),
            "masks": torch.stack(masks),
            "scale_ids": torch.stack(sids),
        },
        out_path,
    )


def preprocess_consistency_features(
    records: list[LabeledSequence],
    config: CanonicalKmerConfig,
    out_path: str | Path,
    num_workers: int | None = None,
    parallel_threshold: int = 32,
) -> None:
    transformed = [
        LabeledSequence(r.sequence_id, _consistency_transform(r, idx), r.mobility, r.amr, r.expansion)
        for idx, r in enumerate(records)
    ]
    preprocess_all_features(transformed, config, out_path, num_workers, parallel_threshold)
    packed = torch.load(out_path, weights_only=True)
    for key in ("features", "struct_features"):
        if key in packed:
            packed[key] = packed[key].half()
    torch.save(packed, out_path)


def extract_features(data_dir: str | Path, config: dict | None = None) -> None:
    data_dir = Path(data_dir)
    kt = config or {}
    feat_cfg = CanonicalKmerConfig(
        window_sizes=tuple(kt.get("window_sizes", (512, 2048, 8192))),
        strides=tuple(kt.get("strides", (256, 1024, 4096))),
        max_windows=_resolve_max_windows(kt.get("max_windows", (16, 8, 4))),
        ngram_min=kt.get("ngram_min", 4),
        ngram_max=kt.get("ngram_max", 6),
        n_structural_features=kt.get("n_structural_features", 19),
        rc_consensus=kt.get("rc_consensus", False),
    )
    expansion_n = kt.get("expansion_classes", 1)
    for name in ("train", "val", "test", "heldout_test", "nonplasmid_control"):
        jsonl_path = data_dir / f"{name}.jsonl"
        if not jsonl_path.exists():
            continue
        records = load_jsonl(jsonl_path)
        if not records:
            continue
        labels = {
            "mobility": torch.tensor([r.mobility for r in records], dtype=torch.long),
            "amr": torch.tensor([float(r.amr) for r in records]),
        }
        if expansion_n > 1:
            labels["expansion"] = torch.tensor(
                [min(int(r.expansion), expansion_n - 1) for r in records], dtype=torch.long
            )
        else:
            labels["expansion"] = torch.tensor([float(r.expansion) for r in records])
        torch.save(labels, data_dir / f"{name}_labels.pt")
        preprocess_all_features(
            records, feat_cfg, data_dir / f"{name}_features.pt", num_workers=kt.get("num_workers", 4)
        )
        if name == "train" and kt.get("build_consistency_cache", False):
            preprocess_consistency_features(
                records, feat_cfg, data_dir / f"{name}_consistency_features.pt", num_workers=kt.get("num_workers", 4)
            )
