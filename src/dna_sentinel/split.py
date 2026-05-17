from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

from dna_sentinel.fasta import canonical_dna


@dataclass(frozen=True)
class SequenceRecord:
    sequence_id: str
    dna: str
    labels: dict[str, Any]


def _kmers(seq: str, k: int) -> set[str]:
    dna = canonical_dna(seq)
    if len(dna) < k:
        return {dna}
    return {dna[i : i + k] for i in range(len(dna) - k + 1)}


def kmer_jaccard(a: str, b: str, k: int = 15) -> float:
    ka, kb = _kmers(a, k), _kmers(b, k)
    if not ka and not kb:
        return 1.0
    overlap = len(ka & kb)
    jaccard = overlap / max(1, len(ka | kb))
    containment = overlap / max(1, min(len(ka), len(kb)))
    return max(jaccard, containment)


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
    vals = []
    for kmer in _sampled_kmers(seq, k):
        digest = hashlib.blake2b(kmer.encode(), digest_size=8).digest()
        vals.append(int.from_bytes(digest, "little"))
    return tuple(sorted(vals)[:n])


def _sketch_similarity(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    overlap = len(sa & sb)
    return max(overlap / max(1, len(sa | sb)), overlap / max(1, min(len(sa), len(sb))))


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
    sketches = [_sketch(rec.dna, k=15, n=48) for rec in records]
    buckets: dict[int, list[int]] = {}
    for i, sketch in enumerate(sketches):
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
    for i, group in enumerate(groups):
        if i < n_test:
            split["test"].extend(group)
        elif i < n_test + n_val:
            split["val"].extend(group)
        else:
            split["train"].extend(group)
    if not split["train"]:
        split["train"] = split["val"][:1]
        split["val"] = split["val"][1:]
    return {k: sorted(v) for k, v in split.items()}
