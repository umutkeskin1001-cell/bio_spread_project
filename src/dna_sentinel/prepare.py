from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from dna_sentinel.dataset import LabeledSequence, save_jsonl
from dna_sentinel.fasta import read_fasta
from dna_sentinel.split import SequenceRecord, cluster_split

MOBILITY_MAP = {"non-mobilizable": 0, "mobilizable": 1, "conjugative": 2}


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
