# BioSpread Sovereign-X Ultra (v4 Frontier)

> **Critical Infrastructure Class, Time-Aware Evidential Learning Pipeline for Global Plasmid Surveillance.**

BioSpread Sovereign-X Ultra is a high-performance system designed to predict the geographic spread of plasmid backbones with clinical-grade reliability. It uses a **Proxy-Based Manifold Alignment** architecture to eliminate cold-start uncertainty and provides rigorous coverage guarantees via **Mondrian Conformal Prediction**.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference API](#inference-api)
- [Configuration](#configuration)
- [Results](#results)
- [Data](#data)
- [Development](#development)
- [Project Structure](#project-structure)
- [Citation](#citation)

---

## Overview

Antimicrobial resistance (AMR) spread via plasmids is a critical public health threat. BioSpread addresses the problem of **predicting geographic spread of plasmid backbones** using:

- **Static features**: plasmid intrinsic properties (size, GC content, mobility, replicon types, etc.)
- **Snapshot / temporal features**: time-varying epidemiological signals (country counts, host diversity, spread velocity, etc.)
- **Taxonomic hierarchy embeddings**: phylum → class → order → family → genus, each with learned embeddings
- **Temporal sequence modeling**: GRU + self-attention over yearly snapshots
- **Multi-horizon hazard prediction**: 1-year, 2-year, and 3-year spread risk

The model is trained with a **leakage-free temporal disjoint split**: backbones observed after the split year (2020) are held out entirely, ensuring no temporal information leakage.

### Key Innovations (Sovereign-X Ultra)

| Feature | Detail |
|---|---|
| **Temporal Proxy Generator** | Imagines missing temporal dynamics from static features via manifold alignment |
| **FiLM Conditioning** | Taxonomy-driven modulation of cold-start features for lineage-aware prediction |
| **Mondrian Conformal** | Group-wise coverage guarantees (90%) for reliable clinical decision support |
| **Evidential Learning** | Direct estimation of epistemic uncertainty to guide automated routing |
| **Hyperbolic Taxonomy** | Poincaré ball embeddings for superior phylogenetic consistency |
| **Unified 4-Loss** | Optimized Multi-Task objective with Adaptive Loss Weighting (Kendall 2018) |

---

## Arch```
                          ┌──────────────────────┐
                          │  Poincaré Taxonomy   │
                          │   (Hyperbolic Ball)  │
                          └──────────┬───────────┘
                                     │ (FiLM)
           ┌─────────────────────────┼──────────────────────────┐
           │                         │                          │
    ┌──────▼──────┐          ┌──────▼──────┐          ┌──────▼──────┐
    │ Static Enc. │          │Temporal Enc.│          │ Proxy Gen.  │
    │ (Gated MLP) │          │ (GRU/Mamba) │          │ (Manifold)  │
    └──────┬──────┘          └──────┬──────┘          └──────┬──────┘
           │                        │                        │
           └─────────┬──────────────┘                (MSE / InfoNCE)
                      │                                      │
               ┌──────▼──────┐                        ┌──────▼──────┐
               │ Confidence-  │◄──────────────────────┤  Cold-Start │
               │ Gated Route  │                       │    Head     │
               └──────┬──────┘                        └─────────────┘
                      │
           ┌──────────┼──────────┐
    ┌──────▼──┐ ┌──────▼──┐ ┌──▼───────┐
    │Evidential│ │Count    │ │Conformal │
    │Hazard H. │ │Head     │ │Wrapper   │
    └──────────┘ └─────────┘ └──────────┘
```
��   (softmax)  │                          │   Head      │
               └──────┬──────┘                          └──────┬──────┘
                      │                                        │
           ┌──────────┼──────────┐                              │
    ┌──────▼──┐ ┌──────▼──┐ ┌──▼───────┐                      │
    │Hazard   │ │Count    │ │Per-timest│                      │
    │Head (h3)│ │Head     │ │Hazard    │                      │
    └─────────┘ └─────────┘ └──────────┘                      │
```

### Model Components

- **TaxonomyEncoder**: 5 separate embedding tables (phylum→genus), each of dimension `taxonomy_embed_dim`, concatenated → dropout
- **StaticEncoder**: MLP with ReLU+Dropout, output `static_dim` (128), with learned gating
- **TemporalEncoder**: Input projection → GRU → self-attention → `temporal_dim` (128)
- **FusionGate**: Softmax over static and temporal representations
- **HazardHead**: MLP → 3-horizon logits
- **CountHead**: 2-layer MLP → log1p(count) prediction
- **TimestepHead**: Per-timestep hazard prediction for ranking loss
- **ColdStartHead**: MLP on static features only (for backbones with no temporal history)

### Loss Function

```
L = λ_bce * L_bce + λ_count * L_count + λ_rank * L_rank 
    + λ_cold * L_cold + λ_all * L_all + λ_gate * L_gate
```

- BCE loss on 3-horizon hazard
- MSE loss on log1p(count) regression
- Ranking loss: within-backbone, later timesteps should have higher hazard
- Cold-start head auxiliary loss
- Per-timestep (all snapshots) hazard loss
- Gate entropy regularization

---

## Installation

### Prerequisites

- Python 3.9+ (tested on 3.9 and 3.11)
- PyTorch 2.0+
- MPS/CUDA recommended for training

### Install

```bash
# Clone the repository
git clone https://github.com/umutgun/bio_spread_project.git
cd bio_spread_project

# Install the package in development mode
pip install -e .

# Optional: install dev dependencies for testing/linting
pip install -e ".[dev]"

# Or using Make
make install
```

### Docker

```bash
docker build -t biospread .
docker run -p 8000:8000 biospread
```

---

## Usage

### Data Preparation

Prepare features from raw plasmid backbone data:

```bash
bio-spread prepare \
    --config config/default.yaml \
    --output-dir data/features
```

This generates:
- `sequences.tsv` — feature matrix with 33 columns
- `taxonomy_vocab.json` — vocabulary mappings for 5 taxonomy levels
- `split.json` — train/val/test backbone IDs (disjoint by backbone)
- `normalizers.npz` — snapshot feature normalizers (mean/std)
- `static_normalizers.npz` — static feature normalizers (mean/std)

### Training

```bash
bio-spread train \
    --config config/default.yaml \
    --feature-dir data/features

# Or using Make
make train
```

Training outputs:
- `artifacts/BS_<timestamp>/best_model.pt` — best model checkpoint
- `artifacts/BS_<timestamp>/metrics.json` — validation metrics
- Platt scalers stored in checkpoint

### Evaluation

```bash
bio-spread evaluate \
    --model-path artifacts/BS_<timestamp>/best_model.pt \
    --config config/default.yaml \
    --feature-dir data/features
```

### Inference API

Start the FastAPI server:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

**Endpoints:**

- `POST /predict` — Predict spread risk for a single backbone
- `GET /health` — Health check

Request format:
```json
{
  "backbone_id": "bb_12345",
  "phylum": "Proteobacteria",
  "class": "Gammaproteobacteria",
  "order": "Enterobacterales",
  "family": "Enterobacteriaceae",
  "genus": "Escherichia",
  "static_features": {
    "log_size": 4.8,
    "gc": 50.7,
    "n_replicon_types": 2,
    "n_relaxase_types": 1,
    "mobility_score": 4.0,
    "is_conjugative": 1,
    "is_mobilizable": 0,
    "topology": 0,
    "n_orit_types": 1,
    "host_range_rank": 3.0
  },
  "snapshots": [
    {
      "year": 2018,
      "n_countries": 2,
      "n_hosts": 3,
      "years_since_first": 0,
      "new_countries_recent": 2,
      "new_countries_2y_ago": 0,
      "n_records": 5,
      "acceleration": 0.0,
      "expansion_ratio": 0.0,
      "niche_breadth": 0.5
    }
  ]
}
```

---

## Configuration

All configurable parameters are in `config/default.yaml` and validated via Pydantic schemas in `src/bio_spread/config/schema.py`.

| Section | Key | Default | Description |
|---|---|---|---|
| **data** | `backbones_path` | `data/.../plasmid_backbones.tsv` | Input plasmid backbone records |
| | `split_year` | 2020 | Temporal split cutoff year |
| | `spread_horizon` | 3 | Prediction horizon in years |
| **model** | `static_dim` | 128 | Static encoder output dimension |
| | `temporal_dim` | 128 | Temporal encoder projection dimension |
| | `gru_hidden` | 192 | GRU hidden size |
| | `gru_layers` | 2 | Number of GRU layers |
| | `taxonomy_embed_dim` | 8 | Per-level embedding dimension |
| **training** | `epochs` | 50 | Maximum training epochs |
| | `patience` | 10 | Early stopping patience |
| | `lr` | 3e-4 | Learning rate |
| | `lambda_count` | 0.15 | Count loss weight |
| | `lambda_rank` | 0.10 | Ranking loss weight |
| | `lambda_cold` | 0.25 | Cold-start loss weight |

---

## Results

### Full Training (50 epochs, validated on 674 backbones)

| Metric | Value |
|---|---|
| **ROC AUC (h3)** | **0.8879** |
| ROC AUC (h1) | 0.9292 |
| ROC AUC (h2) | 0.9231 |
| PR AUC (h3) | 0.7256 |
| F1 Score | 0.6092 |
| Recall | 0.7794 |
| Precision | 0.5000 |
| Balanced Accuracy | 0.7912 |
| Brier Score | 0.1294 |

Full details in [FULL_TRAINING_REPORT.md](./FULL_TRAINING_REPORT.md).

### Temporal Cross-Validation

See `scripts/validate_temporal_cv.py` for expanding-window temporal cross-validation.

---

## Data

Detailed column documentation in [DATA_CODEX.md](./DATA_CODEX.md).

### Input Data

- `data/project_inputs/silver/plasmid_backbones.tsv` — Main backbone records with yearly observations
- `data/external/` — External reference data (host traits, plasmid properties, countries)

### Generated Features

- `data/features/sequences.tsv` — 33-column feature matrix, 21,520 sequences
- `data/features/split.json` — Train: 5,620 / Val: 942 / Test: 279 backbones
- `data/features/taxonomy_vocab.json` — 5-level taxonomy: 35 phyla, 66 classes, 145 orders, 325 families, 915 genera

### Test Fixtures

- `tests/fixtures/leakage/` — Leakage detection test fixtures (6 mock tables)
- `tests/fixtures/geo_holdout.tsv` — Geographic holdout test

---

## Development

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest --cov=src/bio_spread --cov-report=term-missing tests/

# Using Make
make test
```

### Linting

```bash
ruff check src tests
ruff format --check src tests
mypy src
```

### CI/CD

GitHub Actions workflow in `.github/workflows/ci.yml`:
- `fast` job: Ruff + Mypy + tests on Python 3.9 and 3.11
- `full` job: Tests with coverage (≥65% threshold)

---

## Project Structure

```
bio_spread_project/
├── src/
│   └── bio_spread/
│       ├── cli/main.py           # CLI entry point (train, prepare, eval)
│       ├── config/schema.py      # Pydantic config validation
│       ├── data/
│       │   ├── dataset.py        # PyTorch Dataset
│       │   └── snapshot.py       # Feature engineering, sequence building
│       ├── models/
│       │   ├── components.py     # MLP, ColdStartHead
│       │   ├── sovereign.py      # BioSpreadModel architecture
│       │   ├── trainer.py        # Training loop, Platt calibration
│       │   └── __init__.py       # create_model factory
│       └── utils/
│           ├── config.py         # Config loader
│           └── metrics.py        # ROC AUC, PR AUC, Brier, ECE
├── api.py                        # FastAPI inference server
├── scripts/
│   ├── train_baseline_lgb.py     # LightGBM baseline
│   └── validate_temporal_cv.py   # Expanding window CV
├── tests/
│   └── test_redesign.py          # Unit tests (13 tests)
├── config/
│   └── default.yaml              # Default configuration
├── data/
│   └── features/                 # Generated features (gitignored)
├── DATA_CODEX.md                 # Column & data documentation
├── FULL_TRAINING_REPORT.md       # Training results report
├── README.md                     # This file
├── Makefile                      # Convenience commands
├── Dockerfile                    # Docker deployment
└── pyproject.toml                # Project metadata & tool config
```

---

## Citation

If you use BioSpread in your research, please cite:

```bibtex
@software{bio_spread,
  title = {BioSpread: Time-Aware Learning for Plasmid Spread Prediction},
  author = {BioSpread Team},
  year = {2025},
  url = {https://github.com/umutgun/bio_spread_project}
}
```

---

## License

MIT
