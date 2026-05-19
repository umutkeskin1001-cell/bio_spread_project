"""Data preparation and cluster-based splitting module."""
from __future__ import annotations

import hashlib
import json
import random
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from dna_sentinel.utils import LabeledSequence, canonical_dna, read_fasta, save_jsonl

MOBILITY_MAP = {"non-mobilizable": 0, "mobilizable": 1, "conjugative": 2}


# =====================================================================
# CLUSTER SPLITTING UTILITIES (MERGED FROM split.py)
# =====================================================================

@dataclass(frozen=True)
class SequenceRecord:
    sequence_id: str
    dna: str
    labels: dict[str, Any]


def _sampled_kmers(seq: str, k: int, max_kmers: int = 2000) -> list[str]:
    dna = canonical_dna(seq)
    if len(dna) < k:
        return [dna]
    total = len(dna) - k + 1
    if total <= max_kmers:
        return [dna[i : i + k] for i in range(total)]
    step = total / max_kmers
    return [dna[int(i * step) : int(i * step) + k] for i in range(max_kmers)]


def _sketch(seq: str, k: int = 15, n: int = 32) -> tuple[int, ...]:
    vals = [zlib.crc32(kmer.encode()) for kmer in _sampled_kmers(seq, k)]
    return tuple(sorted(vals)[:n])


def _sketch_similarity(sa: set[int], sb: set[int]) -> float:
    if not sa and not sb:
        return 1.0
    overlap = len(sa & sb)
    union_len = len(sa) + len(sb) - overlap
    return max(overlap / max(1, union_len), overlap / max(1, min(len(sa), len(sb))))


class _DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def cluster_split(
    records: list[SequenceRecord],
    seed: int = 42,
    threshold: float = 0.80,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> dict[str, list[str]]:
    if not records:
        return {"train": [], "val": [], "test": []}

    dsu = _DSU(len(records))
    group_seen: dict[str, int] = {}
    exact_seen: dict[str, int] = {}

    for i, rec in enumerate(records):
        group = rec.labels.get("group") or rec.labels.get("backbone_id")
        if group:
            group_key = str(group)
            if group_key in group_seen:
                dsu.union(i, group_seen[group_key])
            else:
                group_seen[group_key] = i
        digest = hashlib.blake2b(canonical_dna(rec.dna).encode(), digest_size=16).hexdigest()
        if digest in exact_seen:
            dsu.union(i, exact_seen[digest])
        else:
            exact_seen[digest] = i

    # Compute MinHash sketches
    if len(records) <= 64:
        raw_sketches = [_sketch(rec.dna, 15, 48) for rec in records]
    else:
        from multiprocessing import Pool
        with Pool() as pool:
            raw_sketches = pool.starmap(_sketch, [(rec.dna, 15, 48) for rec in records])

    sketches = [set(s) for s in raw_sketches]
    buckets: dict[int, list[int]] = {}
    for i, sketch in enumerate(raw_sketches):
        for key in sketch[:12]:
            buckets.setdefault(key, []).append(i)

    seen: set[tuple[int, int]] = set()
    for ids in buckets.values():
        if len(ids) > 256:
            continue
        for pos, a in enumerate(ids):
            for b in ids[pos + 1 :]:
                pair = (min(a, b), max(a, b))
                if pair in seen:
                    continue
                seen.add(pair)
                if _sketch_similarity(sketches[a], sketches[b]) >= threshold:
                    dsu.union(a, b)

    clusters: dict[int, list[str]] = {}
    for i, rec in enumerate(records):
        clusters.setdefault(dsu.find(i), []).append(rec.sequence_id)

    groups = list(clusters.values())
    rng = random.Random(seed)
    rng.shuffle(groups)

    split = {"train": [], "val": [], "test": []}
    n = len(groups)
    n_test = max(1, round(n * test_frac)) if n >= 3 else 1
    n_val = max(1, round(n * val_frac)) if n >= 3 else 1

    for i, gp in enumerate(groups):
        if i < n_test:
            split["test"].extend(gp)
        elif i < n_test + n_val:
            split["val"].extend(gp)
        else:
            split["train"].extend(gp)

    if not split["train"]:
        split["train"] = split["val"][:1]
        split["val"] = split["val"][1:]

    return {k: sorted(v) for k, v in split.items()}


# =====================================================================
# DATASET BUILDER
# =====================================================================

def build_labels(
    backbones_tsv: str | Path,
    amr_tsv: str | Path,
    expansion_country_threshold: int = 15,
) -> dict[str, dict[str, int]]:
    cols = ["sequence_accession", "predicted_mobility", "backbone_id", "country", "resolved_year"]
    bb = pl.read_csv(backbones_tsv, separator="\t", columns=cols)
    amr = pl.read_csv(amr_tsv, separator="\t", columns=["sequence_accession", "amr_any"])
    spread = (
        bb.group_by("backbone_id")
        .agg(pl.col("country").drop_nulls().n_unique().alias("n_countries"))
        .with_columns((pl.col("n_countries") >= expansion_country_threshold).cast(pl.Int64).alias("expansion"))
    )
    joined = (
        bb.join(amr, on="sequence_accession", how="left", coalesce=True)
        .join(spread.select(["backbone_id", "expansion"]), on="backbone_id", how="left")
        .with_columns(
            pl.col("amr_any").fill_null(False).cast(pl.Int64),
            pl.col("expansion").fill_null(0).cast(pl.Int64),
        )
    )
    labels = {}
    for row in joined.to_dicts():
        labels[str(row["sequence_accession"])] = {
            "mobility": MOBILITY_MAP.get(str(row["predicted_mobility"]), 0),
            "amr": int(row["amr_any"]),
            "expansion": int(row["expansion"]),
            "group": str(row["backbone_id"]),
        }
    return labels


def prepare_dataset(
    fasta_path: str | Path,
    backbones_tsv: str | Path,
    amr_tsv: str | Path,
    out_dir: str | Path,
    limit: int = 4096,
    min_len: int = 1000,
    max_len: int = 300_000,
    seed: int = 42,
    expansion_country_threshold: int = 15,
) -> dict[str, int]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels = build_labels(backbones_tsv, amr_tsv, expansion_country_threshold=expansion_country_threshold)
    records: list[LabeledSequence] = []
    seen = set()
    for sid, dna in read_fasta(fasta_path):
        key = sid.split(".")[0]
        lab = labels.get(sid) or labels.get(key)
        if lab is None or sid in seen:
            continue
        if not (min_len <= len(dna) <= max_len):
            continue
        if dna.count("N") / max(1, len(dna)) > 0.05:
            continue
        seen.add(sid)
        records.append(LabeledSequence(sid, dna, lab["mobility"], lab["amr"], lab["expansion"]))
        if len(records) >= limit:
            break

    split = cluster_split(
        [
            SequenceRecord(
                r.sequence_id,
                r.dna,
                {
                    "amr": r.amr,
                    "mobility": r.mobility,
                    "expansion": r.expansion,
                    "group": (labels.get(r.sequence_id) or labels.get(r.sequence_id.split(".")[0]) or {}).get("group", r.sequence_id),
                },
            )
            for r in records
        ],
        seed=seed,
    )
    id_to_record = {r.sequence_id: r for r in records}
    for name, ids in split.items():
        save_jsonl([id_to_record[i] for i in ids if i in id_to_record], out / f"{name}.jsonl")
    (out / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    return {f"{k}_n": len(v) for k, v in split.items()} | {"total_n": len(records)}
