# Cassiopeia Prime Model Card

## Overview

| Property | Value |
|---|---|
| Name | Cassiopeia Prime v2.0.0 |
| Type | Compact multi-task neural model for plasmid risk triage |
| Input | Raw DNA FASTA -> RC-averaged inference over 56 windows x 2,728 canonical k-mer features + 19 structural features |
| Outputs | Mobility class, AMR probability, expansion probability, task-specific evidence windows |
| Training cap | 2,048 plasmids maximum |
| Current train split | 1,414 plasmids |
| Parameters | 501,526 trainable |
| Checkpoint | `artifacts/cassiopeia_prime/cassiopeia_best.pt` |
| Checkpoint size | 5.53 MB |
| Cached CPU latency | 0.70 ms/sequence |
| FASTA CPU latency | 25.01 ms for a 6 kb query |

## Architecture

1. Multi-scale canonical k-mer extraction: 32 x 512 bp, 16 x 2,048 bp, 8 x 8,192 bp windows.
2. Deterministic ternary fast random projection: 2,728 -> 320.
3. Rank-12 F-LoRA correction over the projection.
4. Structural feature fusion for GC, skew, and dinucleotide composition.
5. Circular plasmid positional encoding to reduce linear FASTA cut-point bias.
6. Lightweight window motif convolution over neighboring windows.
7. Three compact GLUMixer blocks with proper DropPath.
8. Task adapters and task-specific evidence pooling.
9. Calibrated task heads for mobility, AMR, and expansion.
10. Reverse-complement averaged deployment for inference-time invariance.

## Data

The training dataset is prepared from the local PLSDB-derived input bundle with group-aware splitting (21-mer, Jaccard >= 0.85). The `config/cassiopeia_prime.yaml` data limit is fixed at 2,048.

| Split | Sequences |
|---|---:|
| Train | 1,414 |
| Validation | 280 |
| Test | 354 |
| Held-out test | 1,200 |
| Non-plasmid control | 900 |

## Audited Performance

| Split | Mobility BA | AMR AUROC | Expansion AUROC | Task Score |
|---:|---:|---:|---:|
| Validation | 77.89% | 86.73% | 85.86% | 83.49% |
| Test | 69.88% | 88.83% | 76.64% | 78.45% |
| Held-out | 79.90% | 93.28% | 83.30% | 85.49% |

### Cross-Validation (5-fold, train+val combined)

| Metric | Mean | Std |
|---:|---:|---:|
| Mobility BA | 80.48% | ± 2.00% |
| AMR AUROC | 93.43% | ± 1.75% |
| Expansion AUROC | 89.13% | ± 2.40% |
| Task Score | 87.68% | ± 0.58% |

### Non-Plasmid Stress Set

All labels are negative. Metrics measure false-positive rates.

| Non-plasmid stress metric | Value |
|---:|---:|
| False mobile rate | 5.58% |
| False AMR rate | 0.33% |
| False expansion rate | 0.00% |
| Mean risk score | — |

## v2.0.0 Changes

| Change | Impact |
|---|---|
| Proper DropPath module | Training dynamics slightly shifted, metrics within expected range |
| Training loop refactor | 75-line loop → 4 functions, each loss logged separately |
| Cross-validation | New `dna-sentinel cross-validate` command, 5-fold reporting |
| Seed management | `set_seed()` centralized, deterministic mode available |
| Logging | `print()` → `logging` module, file + stdout |
| Input validation | API validates DNA length/content, CLI validates config |
| Experiment tracking | `CassiopeiaExperiment` class, automatic history/checkpoint management |
| Test coverage | 85%+ with 114 tests |
| Config validation | Runtime checks for max_windows mismatch, odd kernel |
| Docker | Multi-stage build, non-root user, HEALTHCHECK |

## Intended Use

Cassiopeia Prime is a DNA-only plasmid triage model for education, research prototyping, and comparative benchmarking under a small-data constraint. It is designed to flag mobility, AMR cargo, and spread risk signals quickly from sequence alone.

## Limitations

- Evidence windows are model attention scores, not experimentally validated breakpoints.
- The model does not run BLAST, gene calling, host metadata lookup, or assembly quality checks.
- Non-plasmid false expansion rate (36%) needs improvement.
- RC averaging fixes reverse-complement instability, but circular cut-point drift is reduced rather than eliminated.
- 2,048 plasmid limit means the model is inherently data-constrained.

## Safety

Predictions must not be used as clinical, environmental, regulatory, or biosafety decisions by themselves. Treat outputs as screening signals that need orthogonal validation.
