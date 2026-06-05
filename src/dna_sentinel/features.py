from __future__ import annotations

import functools
import hashlib
import json
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

FEATURE_SCHEMA_VERSION = "v9.0"


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
    n_structural_features: int = 49



def _resolve_max_windows(max_windows: int | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if not isinstance(max_windows, int):
        return tuple(max_windows)
    base = (16, 8, 4)
    total = max(3, max_windows)
    out = [max(1, round(total * b / sum(base))) for b in base]
    while sum(out) != total:
        i = max(range(len(base)), key=lambda j: base[j] if sum(out) < total else out[j])
        if sum(out) < total:
            out[i] += 1
        elif out[i] > 1:
            out[i] -= 1
        else:
            break
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
        self._motif_kmers = {
            "CGATCG": 0, "CGCATG": 1, "CTGCAG": 2, "GATATC": 3, "GAATTC": 4,
            "TACCGG": 5, "TTCCGG": 6, "CCGGTA": 7, "CCGGGA": 8, "CCGGAT": 9,
            "TCCCTG": 10, "CAGGGA": 11, "CATGCC": 12, "CTCGAG": 13, "AGATCT": 14,
            "GGGCCC": 15, "CCTCGG": 16, "CCGAGG": 17, "CCGCGG": 18, "GGATCC": 19,
            "CGAGGG": 20, "CCTCGA": 21, "GCCCTC": 22, "CGACCC": 23, "GAGGGC": 24,
            "CGGGCC": 25, "GCGAGG": 26, "TCGACG": 27, "GGCGCC": 28, "CCCCCG": 29,
        }
        self._n_motifs = 30

    def _struct(self, w: torch.Tensor, lengths: torch.Tensor, ws: int) -> torch.Tensor:
        dev = w.device
        v = (torch.arange(ws, device=dev).unsqueeze(0) < lengths.unsqueeze(1)) & (w >= 0)
        n_base = (w < 0).float()
        wc = w.clamp_min(0)
        g, c, a, t = (wc == 2).float(), (wc == 1).float(), (wc == 0).float(), (wc == 3).float()
        vf = v.float()
        lf = lengths.float().clamp_min(1)

        gc_sum = ((g + c) * vf).sum(dim=1)
        gc = gc_sum / lf
        gs = ((g - c) * vf).sum(dim=1) / gc_sum.clamp_min(1)
        at_s = ((a - t) * vf).sum(dim=1) / ((a + t) * vf).sum(dim=1).clamp_min(1)

        pv = v[:, :-1] & v[:, 1:]
        di = (F.one_hot((wc[:, :-1] * 4 + wc[:, 1:]).long().clamp_max(15), 16).float() * pv.unsqueeze(-1)).sum(dim=1)
        di = di / di.sum(dim=-1, keepdim=True).clamp_min(1)

        n_ratio = (n_base * vf).sum(dim=1) / lf

        N, W = w.shape
        bases_valid = wc * vf.long()
        changes = torch.ones(N, W, dtype=torch.bool, device=dev)
        changes[:, 1:] = bases_valid[:, 1:] != bases_valid[:, :-1]
        changes = changes & v
        run_ids = changes.long().cumsum(dim=1) - 1
        run_ids = run_ids * v.long()
        max_id_per_seq = run_ids.max(dim=1).values
        max_id_global = int(max_id_per_seq.max().item()) + 1 if max_id_per_seq.numel() else 1
        flat_run = run_ids.view(-1)
        batch_idx = torch.arange(N, device=dev).unsqueeze(1).expand_as(run_ids).reshape(-1)
        global_idx = batch_idx * max_id_global + flat_run
        rc_cnt = torch.zeros(N * max_id_global, device=dev)
        rc_cnt.scatter_add_(0, global_idx, torch.ones_like(flat_run, dtype=rc_cnt.dtype))
        rc_cnt = rc_cnt.view(N, max_id_global)
        valid_mask = rc_cnt > 0
        masked = rc_cnt.masked_fill(~valid_mask, -1)
        max_run = torch.where(valid_mask.any(dim=1), masked.max(dim=1).values, torch.zeros(N, device=dev))
        sum_run = torch.where(valid_mask, rc_cnt, 0.0).sum(dim=1)
        mean_run = torch.where(valid_mask.any(dim=1), sum_run / valid_mask.sum(dim=1).clamp_min(1), torch.zeros(N, device=dev))

        base_counts = torch.stack([(ch * vf).sum(dim=1) / lf for ch in (a, c, g, t)], dim=-1)
        freq = base_counts.clamp_min(1e-8)
        entropy = -(freq * freq.log()).sum(dim=-1)
        low_complexity = freq.max(dim=-1).values - 0.25
        len_bucket = (lengths.float() / 1000.0).clamp(0, 5) / 5.0

        return torch.cat([
            gc.unsqueeze(-1), gs.unsqueeze(-1), at_s.unsqueeze(-1),
            di,
            n_ratio.unsqueeze(-1),
            max_run.unsqueeze(-1) / ws, mean_run.unsqueeze(-1) / ws,
            entropy.unsqueeze(-1), low_complexity.unsqueeze(-1),
            len_bucket.unsqueeze(-1),
            base_counts,
        ], dim=-1)

    def _motif_counts(self, w: torch.Tensor) -> torch.Tensor:
        N, ws = w.shape
        out = torch.zeros(N, self._n_motifs, device=w.device)
        bs = self.char_map
        for kmer_str, idx in self._motif_kmers.items():
            k = len(kmer_str)
            if ws < k:
                continue
            pattern = torch.tensor([bs[ord(ch)] for ch in kmer_str], device=w.device, dtype=torch.long)
            windows = w.unfold(1, k, 1)
            matches = (windows == pattern).all(dim=-1).float()
            out[:, idx] = matches.sum(dim=1) / max(1, ws - k + 1)
        return out

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
        if self._n_motifs > 0:
            motifs = self._motif_counts(w)
            struct = torch.cat([struct, motifs], dim=-1)
        ns = self.config.n_structural_features
        if struct.shape[1] > ns:
            struct = struct[:, :ns]
        elif struct.shape[1] < ns:
            struct = F.pad(struct, (0, ns - struct.shape[1]))
        nf = self.n_features
        raw = torch.zeros(N, nf, device=dev)
        mult_d = {k: v.to(dev) for k, v in self._mult.items()}
        cmap_d = {k: v.to(dev) for k, v in self._cmap.items()}
        batch_offset = (torch.arange(N, device=dev).unsqueeze(1) * nf)
        for k in self._km_range:
            m = ws - k + 1
            kmers = w.unfold(-1, k, 1)
            hashes = (kmers * mult_d[k]).sum(dim=-1)
            canon = cmap_d[k][hashes] + self._offsets[k]
            valid = torch.arange(m, device=dev).unsqueeze(0) < (lengths - k + 1).clamp_min(0).unsqueeze(1)
            if not valid.any():
                continue
            raw += (
                torch.bincount((canon + batch_offset)[valid], minlength=N * nf)
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
            out_s = torch.zeros(mw, ns, device=dev)
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
            seq = seq.clamp_min(0)
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
    split_ids = sorted([r.sequence_id for r in records])
    manifest_hash = hashlib.sha256(json.dumps(split_ids).encode()).hexdigest()[:16]
    torch.save(
        {
            "features": torch.stack(feats),
            "struct_features": torch.stack(structs),
            "masks": torch.stack(masks),
            "scale_ids": torch.stack(sids),
            "_schema_version": FEATURE_SCHEMA_VERSION,
            "_n_structural_features": config.n_structural_features,
            "_window_sizes": list(config.window_sizes),
            "_strides": list(config.strides),
            "_max_windows": list(config.max_windows),
            "_n_samples": len(records),
            "_manifest_hash": manifest_hash,
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
        n_structural_features=kt.get("n_structural_features", 49),
        rc_consensus=kt.get("rc_consensus", False),
    )
    expansion_n = kt.get("expansion_classes", 1)
    for name in ("train", "val", "test", "heldout_test"):
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
