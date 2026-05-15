# Data Codex — BioSpread Sovereign-X Pro

> Comprehensive inventory of all data files, column schemas, fixtures, and external resources used in the project.

---

## 1. Sovereign Feature Matrix

**Path:** `data/sovereign_features/sequences.tsv`

**Format:** TSV, 21,520 rows × 33 columns.

**Split:** Train: 13,947 / Val: 5,418 / Test: 2,155 sequences (from 5,620 / 942 / 279 backbones).

### 1.1 Identifier Columns

| Column | Type | Description |
|---|---|---|
| `backbone_id` | str | Unique plasmid backbone identifier |
| `year` | int | Observation year |
| `split` | int | 0=train, 1=val, 2=test (disjoint by backbone) |

### 1.2 Static Features (backbone-intrinsic, time-invariant)

| Column | Type | Description |
|---|---|---|
| `log_size` | float | log₁₀(plasmid size in kb) |
| `gc` | float | GC content percentage |
| `n_replicon_types` | int | Number of distinct replicon types |
| `n_relaxase_types` | int | Number of distinct relaxase types |
| `mobility_score` | float | Mobility score (conjugation efficiency proxy) |
| `is_conjugative` | int | 1 if conjugative, 0 otherwise |
| `is_mobilizable` | int | 1 if mobilizable (but not self-conjugative), 0 otherwise |
| `topology` | int | Plasmid topology encoding (0=linear, 1=circular, etc.) |
| `n_orit_types` | int | Number of oriT types present |
| `host_range_rank` | float | Host range width score |

### 1.3 Snapshot / Temporal Features (time-varying)

| Column | Type | Description |
|---|---|---|
| `n_countries` | int | Number of distinct countries where observed |
| `n_hosts` | int | Number of distinct host genera |
| `years_since_first` | int | Years since first observation of this backbone |
| `new_countries_recent` | int | New countries in the most recent year |
| `new_countries_2y_ago` | int | New countries two years prior |
| `n_records` | int | Total number of records (isolates) |
| `acceleration` | float | Second derivative of country count growth |
| `expansion_ratio` | float | Ratio of new countries to existing countries |
| `spread_velocity` | float | New countries per year since first observation |
| `niche_breadth` | float | Host diversity index (normalized) |

### 1.4 Taxonomy Indices

| Column | Type | Description |
|---|---|---|
| `phylum_idx` | int | Phylum vocabulary index (0 = unknown/padding) |
| `class_idx` | int | Class vocabulary index |
| `order_idx` | int | Order vocabulary index |
| `family_idx` | int | Family vocabulary index |
| `genus_idx` | int | Genus vocabulary index |

**Vocabulary sizes:** 35 phyla, 66 classes, 145 orders, 325 families, 915 genera.
Built from **train-only** data to prevent taxonomy leakage.

### 1.5 Label Columns

| Column | Type | Description |
|---|---|---|
| `hazard_1` | int | 1 if spread to ≥1 new country within 1 year |
| `hazard_2` | int | 1 if spread to ≥1 new country within 2 years |
| `hazard_3` | int | 1 if spread to ≥1 new country within 3 years |
| `n_new_countries` | int | Number of new countries in 3-year horizon (regression target) |
| `observed` | int | 1 if backbone was observed at all in the horizon window |

---

## 2. Split Configuration

**Path:** `data/sovereign_features/split.json`

```json
{
  "train": [5620 backbone IDs],
  "val": [942 backbone IDs],
  "test": [279 backbone IDs]
}
```

**Split strategy:** Temporal disjoint backbone split. Backbones first observed after `split_year` (2020) are assigned to test/val. Remaining backbones are split by fraction (`val_backbone_frac=0.15`, `test_backbone_frac=0.15`). No backbone appears in more than one split.

---

## 3. Taxonomy Vocabulary

**Path:** `data/sovereign_features/taxonomy_vocab.json`

Maps taxonomy labels to integer indices per level:
```json
{
  "TAXONOMY_phylum": {"Proteobacteria": 1, "Firmicutes": 2, ...},
  "TAXONOMY_class": {"Gammaproteobacteria": 1, "Bacilli": 2, ...},
  "TAXONOMY_order": {"Enterobacterales": 1, "Lactobacillales": 2, ...},
  "TAXONOMY_family": {"Enterobacteriaceae": 1, "Streptococcaceae": 2, ...},
  "genus": {"Escherichia": 1, "Streptococcus": 2, ...}
}
```

Index 0 is reserved for unknown/padding.

---

## 4. Normalizers

**Path:** `data/sovereign_features/normalizers.npz`
- `means`: float32 array, shape (10,) — snapshot feature means
- `stds`: float32 array, shape (10,) — snapshot feature stds

**Path:** `data/sovereign_features/static_normalizers.npz`
- `means`: float32 array, shape (10,) — static feature means
- `stds`: float32 array, shape (10,) — static feature stds

---

## 5. External Data

### 5.1 Plasmid Intrinsic Properties

**Path:** `data/external/plasmid_intrinsic_props.tsv`

| Column | Description |
|---|---|
| `backbone_id` | Unique backbone identifier |
| `size_kb` | Plasmid size in kilobases |
| `gc_content` | GC content percentage |
| `replicon_types` | Comma-separated replicon types |
| `relaxase_types` | Comma-separated relaxase types |
| `mobility` | Mobility classification (conjugative/mobilizable/non-mobilizable) |
| `topology` | Circular or linear |
| `orit_types` | oriT types present |
| `conjugation_score` | Conjugation efficiency score |
| `toxin_antitoxin_count` | Number of TA systems |

### 5.2 Host Traits

**Path:** `data/external/host_traits.tsv`

| Column | Description |
|---|---|
| `genus` | Host genus name |
| `family` | Host family |
| `class` | Host class |
| `phylum` | Host phylum |
| `is_pathogen` | 1 if known pathogen |
| `environment_primary` | Primary environment (clinical, water, soil, etc.) |
| `gram_stain` | Gram stain classification |
| `metabolism` | Metabolic type |

### 5.3 Countries

**Path:** `data/external/countries.json`

Mapping of country codes to country names.

---

## 6. Input Data (Project Inputs)

**Path:** `data/project_inputs/silver/plasmid_backbones.tsv`

The main input table. Contains yearly backbone observation records with columns including:
- `backbone_id`, `year`, `country`, `host_genus`
- `phylum`, `class`, `order`, `family`, `genus` (host taxonomy)
- `amr_gene_count` (AMR gene count per record)
- Plasmid intrinsic properties (size, GC, replicons, relaxases, mobility, etc.)

**Note:** This file is gitignored. See `data/project_inputs/README.md` for full provenance.

---

## 7. Sample Data

| File | Description |
|---|---|
| `data/sample_records.tsv` | Small sample of backbone records for quick testing |
| `data/sample_plasmid_records.csv` | CSV variant of sample records |

---

## 8. Test Fixtures

### 8.1 Leakage Detection Fixtures

Located in `tests/fixtures/leakage/`. Used by `tests/test_redesign.py` to verify the pipeline correctly rejects leakage scenarios.

| File | Description |
|---|---|
| `records_with_future.csv` | Records that simulate test-set backbones appearing in training data |
| `amr_mock.tsv` | Mock AMR gene data for AMR feature testing |
| `host_traits_mock.tsv` | Mock host taxonomy data (4 hosts) |
| `plasmid_intrinsic_props_mock.tsv` | Mock plasmid properties (2 backbones) |
| `country_indicators_mock.csv` | Mock country-level indicators (up to 2020) |
| `geo_leak.tsv` | Geographic leakage scenario: labels computed using test-country data |

### 8.2 Geographic Holdout

| File | Description |
|---|---|
| `tests/fixtures/geo_holdout.tsv` | Geographic holdout test data |

---

## 9. Config Files

### 9.1 Default Config

**Path:** `config/default.yaml`

Full training configuration with all hyperparameters (50+ parameters organized in `data`, `model`, `training` sections). Validated by Pydantic schema at `src/bio_spread_reborn/config/schema.py`.

### 9.2 Schema

**Path:** `src/bio_spread_reborn/config/schema.py`

Three Pydantic models:
- `DataConfig` — data paths, split parameters, horizon
- `ModelConfig` — architecture dimensions and hyperparameters
- `TrainingConfig` — optimizer, loss weights, regularization
- `Config` — top-level container

---

## 10. Artifacts

**Path:** `artifacts/SX_<timestamp>/`

| File | Description |
|---|---|
| `best_model.pt` | PyTorch model checkpoint (state_dict + Platt scalers) |
| `metrics.json` | Validation metrics (ROC AUC, PR AUC, Brier, ECE, F1, Precision, Recall, etc.) |

---

## 11. Column Type Summary

| Category | Columns | Count |
|---|---|---|
| Identifiers | `backbone_id`, `year`, `split` | 3 |
| Static features | `log_size`, `gc`, `n_replicon_types`, `n_relaxase_types`, `mobility_score`, `is_conjugative`, `is_mobilizable`, `topology`, `n_orit_types`, `host_range_rank` | 10 |
| Snapshot features | `n_countries`, `n_hosts`, `years_since_first`, `new_countries_recent`, `new_countries_2y_ago`, `n_records`, `acceleration`, `expansion_ratio`, `spread_velocity`, `niche_breadth` | 10 |
| Taxonomy indices | `phylum_idx`, `class_idx`, `order_idx`, `family_idx`, `genus_idx` | 5 |
| Labels | `hazard_1`, `hazard_2`, `hazard_3`, `n_new_countries`, `observed` | 5 |
| **Total** | | **33** |
