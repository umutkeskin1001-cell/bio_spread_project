from __future__ import annotations

import json
from pathlib import Path
import polars as pl
import numpy as np

def build_gene_sharing_edges(
    backbone_ids: list[str],
    raw_dir: Path,
    config_path: Path,
    weight: float = 1.5
) -> list[tuple[str, str, float]]:
    """Builds edges between backbones that share high-priority AMR genes."""
    try:
        # 1. Load high-priority genes
        with open(config_path, "r") as f:
            config = json.load(f)
        hp_genes = set()
        for cat in config.get("high_priority_genes", {}).values():
            hp_genes.update(cat.get("genes", []))

        # 2. Load AMR hits and Backbone mapping
        amr_path = raw_dir / "amr.tsv"
        bb_path = raw_dir / "plasmid_backbones.tsv"
        if not (amr_path.exists() and bb_path.exists()):
            return []

        amr_df = pl.read_csv(amr_path, separator="\t")
        bb_df = pl.read_csv(bb_path, separator="\t")

        # Map Accession -> Backbone ID
        acc_to_bb = dict(zip(bb_df["sequence_accession"], bb_df["backbone_id"]))
        
        # Filter AMR hits to high-priority genes and known backbones
        amr_filtered = amr_df.filter(
            pl.col("gene_symbol").is_in(hp_genes)
        ).select(["NUCCORE_ACC", "gene_symbol"]).unique()

        # Group by gene to find backbones sharing them
        gene_to_bbs = {}
        for row in amr_filtered.to_dicts():
            acc = row["NUCCORE_ACC"]
            gene = row["gene_symbol"]
            bb = acc_to_bb.get(acc)
            if bb and bb in backbone_ids:
                if gene not in gene_to_bbs:
                    gene_to_bbs[gene] = set()
                gene_to_bbs[gene].add(bb)

        # 3. Create edges
        edges = []
        target_bb_set = set(backbone_ids)
        for gene, bbs in gene_to_bbs.items():
            bbs_list = sorted(list(bbs))
            # Only connect if the gene is somewhat rare/specific (don't connect 1000 backbones)
            if 1 < len(bbs_list) < 50: 
                for i in range(len(bbs_list)):
                    for j in range(i + 1, len(bbs_list)):
                        edges.append((bbs_list[i], bbs_list[j], weight))
        
        return edges
    except Exception as e:
        print(f"Warning: Failed to build gene-sharing edges: {e}")
        return []
