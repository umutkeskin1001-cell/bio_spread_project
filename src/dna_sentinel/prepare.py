from __future__ import annotations

import heapq
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mmh3
import polars as pl

from dna_sentinel.utils import LabeledSequence, read_fasta, save_jsonl

MOBILITY_MAP = {"non-mobilizable": 0, "mobilizable": 1, "conjugative": 2}
_KMER_K = 21
_KMER_N = 2000
_JACCARD_THRESHOLD = 0.85
_BUCKET_TOPK = 12


@dataclass(frozen=True)
class SequenceRecord:
    sequence_id: str
    dna: str
    labels: dict[str, Any]


def _sampled_kmers(seq: str, k: int = _KMER_K, n: int = _KMER_N) -> set[str]:
    slen = len(seq)
    if slen < k:
        return {seq}
    total = slen - k + 1
    if total <= n:
        return {seq[i:i + k] for i in range(total)}
    step = total / n
    return {seq[int(i * step):int(i * step) + k] for i in range(n)}


def _jaccard(a: set[str], b: set[str]) -> float:
    overlap = len(a & b)
    return overlap / max(1, len(a) + len(b) - overlap)


class _DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _report_similarity(
    records: list[SequenceRecord], split: dict[str, list[str]], sketches: list[set[str]] | None = None,
    threshold: float = _JACCARD_THRESHOLD,
) -> dict:
    rid_to_idx = {r.sequence_id: i for i, r in enumerate(records)}
    if sketches is None:
        sketches = [_sampled_kmers(r.dna) for r in records]
    max_sim = 0.0
    pairs = []
    for name, ids in split.items():
        if name == "train":
            continue
        for sid in ids:
            i = rid_to_idx.get(sid)
            if i is None:
                continue
            for train_sid in split.get("train", []):
                j = rid_to_idx.get(train_sid)
                if j is None:
                    continue
                sim = _jaccard(sketches[i], sketches[j])
                if sim > max_sim:
                    max_sim = sim
                if sim >= threshold:
                    pairs.append({"test": sid, "train": train_sid, "jaccard": round(sim, 4)})
    pairs = sorted(pairs, key=lambda x: -x["jaccard"])[:10]
    return {"max_train_test_jaccard": round(max_sim, 4), "top_similar_pairs": pairs}


def cluster_split(
    records: list[SequenceRecord],
    seed: int = 42,
    threshold: float = _JACCARD_THRESHOLD,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> tuple[dict[str, list[str]], list[set[str]], dict[str, int]]:
    n = len(records)
    if n == 0:
        return {"train": [], "val": [], "test": []}, [], {}

    dsu = _DSU(n)
    group_seen: dict[str, int] = {}
    exact_hashes: dict[int, int] = {}

    for i, rec in enumerate(records):
        g = rec.labels.get("group") or rec.labels.get("backbone_id")
        if g:
            gk = str(g)
            if gk in group_seen:
                dsu.union(i, group_seen[gk])
            group_seen[gk] = i
        h = mmh3.hash(rec.dna[:1000], seed=42)
        if h in exact_hashes:
            dsu.union(i, exact_hashes[h])
        exact_hashes[h] = i

    sketches = [_sampled_kmers(rec.dna) for rec in records]
    buckets: dict[int, list[int]] = {}
    for i, sk in enumerate(sketches):
        for h in [mmh3.hash(x, seed=42) for x in heapq.nsmallest(_BUCKET_TOPK, sk)]:
            buckets.setdefault(h, []).append(i)

    seen_pairs: set[tuple[int, int]] = set()
    for ids in buckets.values():
        if len(ids) > 256:
            continue
        for pi, a in enumerate(ids):
            for b in ids[pi + 1 :]:
                pair = (a, b) if a < b else (b, a)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    if _jaccard(sketches[a], sketches[b]) >= threshold:
                        dsu.union(a, b)

    clusters: dict[int, list[str]] = {}
    for i, rec in enumerate(records):
        clusters.setdefault(dsu.find(i), []).append(rec.sequence_id)

    groups = list(clusters.values())
    rng = random.Random(seed)
    rng.shuffle(groups)

    ng = len(groups)
    if ng < 3:
        if ng == 1:
            return {"train": groups[0], "val": [], "test": []}, sketches, clusters
        split = {"train": groups[0], "val": groups[1] if ng > 1 else [], "test": []}
        return {k: sorted(v) for k, v in split.items()}, sketches, clusters
    n_test = max(1, round(ng * test_frac))
    n_val = max(1, round(ng * val_frac))
    split = {"train": [], "val": [], "test": []}
    for i, gp in enumerate(groups):
        if i < n_test:
            split["test"].extend(gp)
        elif i < n_test + n_val:
            split["val"].extend(gp)
        else:
            split["train"].extend(gp)
    return {k: sorted(v) for k, v in split.items()}, sketches, clusters


def build_labels(
    backbones_tsv: str | Path, amr_tsv: str | Path, expansion_country_threshold: int = 15
) -> dict[str, dict[str, int | str | float]]:
    cols = ["sequence_accession", "predicted_mobility", "backbone_id", "country", "resolved_year"]
    bb = pl.read_csv(backbones_tsv, separator="\t", columns=cols)
    amr = pl.read_csv(amr_tsv, separator="\t", columns=["sequence_accession", "amr_any"])
    agg = bb.group_by("backbone_id").agg(
        pl.col("country").drop_nulls().n_unique().alias("n_countries"),
        pl.col("resolved_year").min().alias("min_year"),
    ).with_columns((pl.col("n_countries") >= expansion_country_threshold).cast(pl.Int64).alias("expansion"))
    joined = (
        bb.join(amr, on="sequence_accession", how="left", coalesce=True)
        .join(agg.select(["backbone_id", "expansion", "min_year"]), on="backbone_id", how="left")
        .with_columns(
            pl.col("amr_any").fill_null(False).cast(pl.Int64),
            pl.col("expansion").fill_null(0).cast(pl.Int64),
            pl.col("resolved_year").cast(pl.Int64).fill_null(0),
        )
    )
    return {
        str(r["sequence_accession"]): {
            "mobility": MOBILITY_MAP.get(str(r["predicted_mobility"]), 0),
            "amr": int(r["amr_any"]),
            "expansion": int(r["expansion"]),
            "group": str(r["backbone_id"]),
            "year": int(r["resolved_year"]),
        }
        for r in joined.to_dicts()
    }


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
    rng = random.Random(seed)
    candidates, seen = [], set()
    for sid, dna in read_fasta(fasta_path):
        key = sid.split(".")[0]
        lab = labels.get(sid) or labels.get(key)
        if lab is None or sid in seen:
            continue
        if not (min_len <= len(dna) <= max_len) or dna.count("N") / len(dna) > 0.05:
            continue
        seen.add(sid)
        candidates.append(LabeledSequence(sid, dna, lab["mobility"], lab["amr"], lab["expansion"]))
    rng.shuffle(candidates)
    records = candidates[:limit]

    seq_records = [
        SequenceRecord(
            r.sequence_id,
            r.dna,
            {
                "amr": r.amr,
                "mobility": r.mobility,
                "expansion": r.expansion,
                "group": labels.get(r.sequence_id.split(".")[0], {}).get("group", r.sequence_id),
            },
        )
        for r in records
    ]
    split, sketches, clusters = cluster_split(seq_records, seed=seed)
    similarity = _report_similarity(seq_records, split, sketches)

    id_to_record = {r.sequence_id: r for r in records}
    for name, ids in split.items():
        save_jsonl([id_to_record[i] for i in ids], out / f"{name}.jsonl")
    total_split = sum(len(ids) for ids in split.values())
    if total_split > 2048:
        raise ValueError(
            f"Data boundary violation: split total {total_split} > 2048. "
            f"Train={len(split.get('train', []))}, Val={len(split.get('val', []))}, Test={len(split.get('test', []))}"
        )
    cluster_map = {}
    for gid, ids in clusters.items():
        for sid in ids:
            cluster_map[sid] = gid
    report_data = {
        **split,
        "_similarity_report": similarity,
        "_cluster_map": cluster_map,
    }
    (out / "split.json").write_text(
        json.dumps(report_data, indent=2), encoding="utf-8"
    )
    return {f"{k}_n": len(v) for k, v in split.items()} | {
        "total_n": len(records),
        "max_similarity": similarity["max_train_test_jaccard"],
    }
