# Cassiopeia Prime Model Card

## Overview

| Property | Value |
|---|---|
| Name | Cassiopeia Prime v2.0.0 |
| Type | Compact multi-task neural model for plasmid risk triage |
| Input | Raw DNA FASTA → RC-averaged inference over 56 windows × 2,728 canonical k-mer features + 49 structural features |
| Outputs | Mobility class, AMR probability, expansion probability, task-specific evidence windows, biological interpretation with confidence labels |
| Training cap | 2,048 plasmids maximum |
| Current train split | 1,414 plasmids |
| Parameters | 568,437 trainable |
| Checkpoint | `artifacts/cassiopeia_prime_v14/cassiopeia_best.pt` |
| Checkpoint size | 5.81 MB |
| Cached CPU latency | 0.59 ms/sequence |
| FASTA CPU latency | 105.5 ms for a 6 kb query (MPS) |

## Architecture

1. Multi-scale canonical k-mer extraction: 32 × 512 bp, 16 × 2,048 bp, 8 × 8,192 bp windows.
2. Deterministic ternary fast random projection: 2,728 → 320.
3. Rank-12 F-LoRA correction over the projection.
4. Structural feature fusion for GC, skew, and dinucleotide composition (49 dims).
5. Circular plasmid positional encoding (CPPE) to reduce linear FASTA cut-point bias.
6. Lightweight window motif convolution over neighboring windows (RingSSM kernel=7).
7. Three compact GLUMixer blocks with proper DropPath (rate=0.12).
8. Task adapters (rank 12) and task-specific evidence pooling.
9. Calibrated task heads for mobility, AMR, and expansion.
10. Reverse-complement averaged deployment for inference-time invariance.

## Data

Training dataset prepared from PLSDB v2025 with group-aware splitting (21-mer, Jaccard ≥ 0.85). Config data limit is fixed at 2,048.

| Split | Sequences |
|---|---:|
| Train | 1,466 |
| Validation | 286 |
| Test | 296 |
| Held-out test | 1,200 |

### Class Distribution

| Split | Total | Mob0 | Mob1 | Mob2 | AMR+ | Exp+ |
|---|---:|---:|---:|---:|---:|---:|
| Train | 1,466 | 575 | 440 | 451 | 613 | 518 |
| Validation | 286 | 110 | 92 | 84 | 113 | 90 |
| Test | 296 | 113 | 102 | 81 | 114 | 93 |
| Held-out | 1,200 | 411 | 320 | 469 | 672 | 542 |

## Audited Performance

| Split | Mobility BA | AMR AUROC | Expansion AUROC | Task Score |
|---:|---:|---:|---:|---:|
| Validation | 75.02% | 90.76% | 87.15% | 84.31% |
| Test | 75.22% | 92.14% | 86.09% | 84.48% |
| **Held-out** | **76.92%** | **93.91%** | **87.01%** | **85.95%** |

### Per-class Mobility F1 (held-out)

| Class | F1 |
|---:|---:|
| non-mobilizable (0) | 0.739 |
| mobilizable (1) | 0.678 |
| conjugative (2) | 0.897 |

### Bootstrap 95% CI (held-out, 500 resamples)

| Metric | Point | CI 95% |
|---|---|---:|---:|
| Mobility BA | 76.92% | 74.63–79.31% |
| AMR AUROC | 93.91% | 92.54–95.15% |
| Expansion AUROC | 87.01% | 85.02–88.93% |
| Task Score | 85.95% | 85.18–86.75% |

### Calibration (ECE — Expected Calibration Error)

| Split | Mobility ECE | AMR ECE | Expansion ECE |
|---|---:|---:|---:|
| Validation | 0.085 | 0.040 | 0.084 |
| Test | 0.112 | 0.036 | 0.069 |
| **Held-out** | **0.116** | **0.064** | **0.044** |

All tasks calibrated via L-BFGS temperature + bias scaling on validation logits. ECE < 0.12 across all splits and tasks.

## Ablation Summary

| Intervention | Held-out Task Score | Δ |
|---|---:|---:|
| **Before** (baseline v14) | 84.97% | — |
| Focal loss γ=0.0→0.5 | 85.42% | +0.45 |
| Consistency weight tuning | 85.63% | +0.21 |
| Calibration (L-BFGS) | 85.95% | +0.32 |
| **Final** | **85.95%** | **+0.98** |

No individual task dropped >2 points relative to baseline.

## CLI

| Command | Alias | Description |
|---|---|---|
| `dna-sentinel train` | `dna train` | Train model |
| `dna-sentinel predict` | `dna predict` | Predict on FASTA |
| `dna-sentinel benchmark` | `dna bench` | Benchmark all splits |
| `dna-sentinel prepare` | `dna prep` | Build dataset |
| `dna-sentinel prepare-features` | `dna features` | Extract features |
| `dna-sentinel cross-validate` | `dna cv` | k-fold CV |
| `dna-sentinel serve` | `dna serve` | Start API server |
| `dna-sentinel experiment-list` | `dna list` | List experiments |
| — | `dna interpret` | Predict with biological interpretation |

All commands accept short flags (e.g., `-m` for checkpoint, `-f` for FASTA, `-c` for config).

## Biological Interpretation

Each prediction includes:
- **Mobility**: Class label + description + confidence (HIGH ≥ 0.80, MEDIUM 0.60–0.79, LOW < 0.60)
- **AMR**: Probability + confidence + CARD family matching when confidence is HIGH
- **Expansion**: Probability + confidence + synthetic reasoning based on mobility/AMR co-occurrence
- **Disclaimer**: "Tarama sinyalidir; klinik, çevresel veya biyogüvenlik kararlarında tek başına kullanılamaz."

## Web Interface

Static HTML/CSS/JS served from `web/`:
- **Prediction page** (`index.html`): FASTA/text input → mobility/AMR/expansion scores + interpretation
- **Benchmark page** (`benchmark.html`): Cassiopeia vs baselines + methodology + limitations

No external API calls — fully offline.

## v2.0.0 → v0.3.0 Changes

| Change | Impact |
|---|---|
| `dna` short alias CLI group | Faster command entry, all `--help` complete |
| Short flags (-m, -f, -c, -j, -i) | Reduces typing for common operations |
| Biological interpretation | Confidence labels, CARD matching, disclaimer |
| Web UI (prediction + benchmark) | Interactive offline demo |
| Focal loss for mobility (γ=0.5) | Improved class 1 recall by ~2% |
| Coverage ≥85% | 371 tests, 88.57% coverage |
| Ablation documentation | Before/after table generated |
| Benchmark methodology | Documented splits, protocol, baseline sources |

## Test Coverage

| Module | Coverage |
|---|---:|
| `__init__.py` | 100% |
| `api.py` | 84% |
| `cli.py` | 70% |
| `features.py` | 88% |
| `model.py` | 98% |
| `prepare.py` | 78% |
| `train.py` | 98% (utility functions; training loop excluded as GPU-dependent) |
| `utils.py` | 95% |
| **Total** | **89.47%** |

## Intended Use

Cassiopeia Prime is a DNA-only plasmid triage model for education, research prototyping, and comparative benchmarking under a small-data constraint. It flags mobility, AMR cargo, and spread risk signals quickly from sequence alone.

## Limitations

- Evidence windows are model attention scores, not experimentally validated breakpoints.
- The model does **not** run BLAST, gene calling, host metadata lookup, or assembly quality checks.
- RC averaging reduces reverse-complement instability but circular cut-point drift is reduced rather than eliminated.
- 2,048-plasmid limit means the model is inherently data-constrained.
- Mobility class 1 (mobilizable) remains the most challenging class (F1 ≈ 0.68).
- Geographic expansion is a proxy based on ≥15 observed countries; sampling bias affects this.

## Safety

Predictions must not be used as clinical, environmental, regulatory, or biosafety decisions by themselves. Treat outputs as screening signals that need orthogonal validation.
