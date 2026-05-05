from __future__ import annotations

import json
from pathlib import Path

import polars as pl


FEATURE_SURFACE = Path("reports/latest_refresh/features.csv")
LINEAGE_PATH = Path("project_config/config/feature_lineage.json")

EXCLUDED = {
    "backbone_id",
    "label_geo_spread",
    "n_new_countries_future",
    "region",
    "knownness_score",
}


def infer_source(col: str) -> str:
    if col.startswith("psge_") or col.startswith("gnn_embed_") or col.startswith("fastrp_"):
        return "graph_embedding"
    if col.startswith("synergy_"):
        return "interaction_transform"
    if col.startswith("oof_"):
        return "stacking_oof"
    if col == "phylo_prop_risk":
        return "phylo_propagation"
    if col in {"grps"}:
        return "embedding_knn"
    if "country" in col or "region" in col:
        return "geo_features"
    if "amr" in col or "carbapenemase" in col or "esbl" in col or "colistin" in col:
        return "amr_features"
    if "host" in col or "pathogenic" in col or "metabolic" in col or "gram" in col:
        return "host_features"
    if col in {"inc_group_code", "mob_typer_code", "conjugation_score", "toxin_antitoxin_count", "replicon_count", "avg_gc_content", "plasmid_size_kb", "phage_related_genes_count"}:
        return "intrinsic_plasmid_features"
    return "geo_surface"


def bootstrap_lineage() -> Path:
    if not FEATURE_SURFACE.exists():
        raise FileNotFoundError(f"Feature surface not found: {FEATURE_SURFACE}")
    df = pl.read_csv(FEATURE_SURFACE, n_rows=1)
    cols = [c for c in df.columns if c not in EXCLUDED]
    features: dict[str, dict[str, object]] = {}
    for col in sorted(cols):
        features[col] = {
            "source": infer_source(col),
            "temporal_scope": "pre_split" if "future" not in col.lower() else "forbidden",
            "label_touched": False,
        }
    payload = {"features": features}
    LINEAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LINEAGE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return LINEAGE_PATH


if __name__ == "__main__":
    path = bootstrap_lineage()
    print(path)
