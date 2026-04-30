# BioSpread Data Contracts

## Raw observation CSV contract

`--mode input --input <csv>` requires these columns:

- `backbone_id`
- `year`
- `country`
- `host_genus`
- `clinical_context`
- `amr_gene_count`
- `mobility_score`

Invalid years, empty backbone identifiers, or non-numeric AMR and mobility
values fail fast.

## Raw packaged tables

`--mode raw` expects:

- `data/raw/plasmid_backbones.tsv`
- `data/raw/amr.tsv`

The runtime derives:

- backbone identifier
- year
- country
- host genus
- clinical context
- AMR count
- mobility score

## GeoSpread feature surface

`--mode geo` expects a TSV containing:

- required base columns:
  `backbone_id`, `spread_label`, `n_new_countries`
- required model feature columns:
  `T_eff_norm`, `H_obs_specialization_norm`, `A_eff_norm`,
  `coherence_score`, `backbone_purity_norm`, `assignment_confidence_norm`,
  `mash_neighbor_distance_train_norm`, `orit_support`,
  `H_external_host_range_norm`
- optional derived geography feature columns:
  `geo_country_entropy_train`, `geo_macro_region_entropy_train`,
  `geo_dominant_region_share_train`, `geo_country_record_count_train`

When optional geography columns are absent, the runtime derives them from
packaged aliases such as `log1p_n_countries_train`, `n_train_macro_regions`,
and `log1p_member_count_train`.

The runtime rejects:

- empty surfaces
- unlabeled surfaces
- surfaces missing required model feature columns

Blocked leakage tokens include `label`, `future`, `test`, `n_new`,
`new_countries`, `event_within`, `time_to`, `visibility`, `outcome`, and
`severity`. These tokens are permitted for labels and retrospective reporting
columns, but not for model input features.

## Artifact interpretation contract

- `audit.json` is the primary reliability and reproducibility artifact.
- `benchmark.json` is the drift-comparison input artifact.
- `trend_report.json` tracks registry-history regressions.
- `release_gate.json` is the operational decision artifact.
- `manifest.json` is the provenance and path-selection artifact.
