#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'data' / 'project_inputs' / 'geo_spread' / 'inputs' / 'external_holdout.tsv'
DST = ROOT / 'data' / 'project_inputs' / 'geo_spread' / 'inputs' / 'external_holdout_curated_v1.tsv'

REQUIRED = [
    'backbone_id',
    'spread_label',
    'n_new_countries',
    'metadata_support_depth_norm',
    'metadata_missingness_burden',
    'assignment_confidence_norm',
    'backbone_purity_norm',
]

def main() -> int:
    df = pl.read_csv(SRC, separator='\t')
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f'missing required columns: {missing}')
    out = df.filter(pl.col('spread_label').is_not_null())
    out = out.unique(subset=['backbone_id'], keep='first')
    out = out.filter(
        pl.col('metadata_support_depth_norm').is_not_null()
        & pl.col('metadata_missingness_burden').is_not_null()
        & (pl.col('metadata_missingness_burden') <= 0.90)
        & pl.col('assignment_confidence_norm').is_not_null()
        & pl.col('backbone_purity_norm').is_not_null()
    )
    out = out.sort('backbone_id')
    DST.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(DST, separator='\t')
    print(f'wrote {DST} rows={out.height} cols={len(out.columns)}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
