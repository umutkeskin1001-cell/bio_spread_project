# Cassiopeia Prime Model Card

## Overview

| Property | Value |
|---|---|
| Name | Cassiopeia Prime v0.3.0 (champion: v14+v15 ensemble) |
| Type | Compact multi-task neural model for plasmid risk triage |
| Input | Raw DNA FASTA → RC-averaged inference over 56 windows × 2,728 canonical k-mer features + 49 structural features |
| Outputs | Mobility class, AMR probability, expansion probability, task-specific evidence windows, biological interpretation with confidence labels |
| Training cap | 2,048 plasmids maximum |
| Current train split | 1,466 plasmids |
| Parameters | 568,437 trainable (× 2 for ensemble) |
| Checkpoints | `artifacts/cassiopeia_prime_v15/cassiopeia_best.pt` + `artifacts/cassiopeia_prime_v14/cassiopeia_best.pt` |
| Ensemble weights | v15 0.47 / v14 0.53 |
| Cached CPU latency | 0.59 ms/sequence |
| FASTA CPU latency | 59.5 ms for a 6 kb query (MPS) |

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
11. **Ensemble**: weighted average of v14 and v15 probability outputs (optimal w=0.47/0.53).

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
|---|---|---:|---:|---:|---:|---:|---:|
| Train | 1,466 | 575 | 440 | 451 | 613 | 518 |
| Validation | 286 | 110 | 92 | 84 | 113 | 90 |
| Test | 296 | 113 | 102 | 81 | 114 | 93 |
| Held-out | 1,200 | 411 | 320 | 469 | 672 | 542 |

## Audited Performance (Ensemble)

| Split | Mobility BA | AMR AUROC | Expansion AUROC | Task Score |
|---:|---:|---:|---:|---:|
| Validation | 77.63% | 91.99% | 88.54% | 86.05% |
| Test | 74.18% | 92.16% | 87.43% | 84.59% |
| **Held-out** | **78.33%** | **93.89%** | **88.78%** | **87.00%** |

> **Inference mode:** All numbers in this table are produced by `dna-sentinel benchmark`
> with ensemble (`--ensemble-checkpoint`), evaluating pre-cached features without
> reverse-complement averaging. This matches the protocol used during model selection
> and is fully reproducible from the committed feature cache (`data/dna_sentinel/*_features.pt`).
> Production-style reverse-complement-averaged numbers (which match the inference path
> used by `dna predict` / `dna interpret`) are ~1–2 points higher on the held-out split
> and can be regenerated with `dna-sentinel benchmark --rc-average --ensemble-checkpoint ...`.
> Reports include an `inference_mode` field (`cached_features` or `rc_averaged`).

### Per-class Mobility F1 (held-out)

| Class | F1 |
|---:|---:|
| non-mobilizable (0) | 0.741 |
| mobilizable (1) | **0.725** |
| conjugative (2) | 0.878 |

### Bootstrap 95% CI (held-out, 500 resamples)

| Metric | Point | CI 95% |
|---|---|---:|---:|
| Mobility BA | 78.33% | 75.92–80.67% |
| AMR AUROC | 93.89% | 92.58–95.04% |
| Expansion AUROC | 88.78% | 87.00–90.55% |
| Task Score | 87.00% | 86.20–87.78% |

### Calibration (ECE — Expected Calibration Error)

| Split | Mobility ECE | AMR ECE | Expansion ECE |
|---|---|---:|---:|---:|
| Validation | 0.148 | 0.061 | 0.055 |
| Test | 0.135 | 0.055 | 0.052 |
| **Held-out** | **0.164** | **0.051** | **0.038** |

All tasks calibrated via L-BFGS temperature + bias scaling on validation logits (applied per model before ensemble averaging).

## Ablation Summary

| Intervention | Held-out Task Score | Δ |
|---|---|---:|---:|
| **Before** (baseline, no calibration) | 84.97% | — |
| + L-BFGS temperature + bias calibration | 85.42% | +0.45 |
| + Consistency weight tuning (0.25) | 85.63% | +0.21 |
| + Final calibration re-fit on val | 85.95% | +0.32 |
| + Per-class weight + focal γ=0.5 (v15) | 85.86% | +0.89 |
| + **v14+v15 ensemble (w=0.47/0.53)** | **87.00%** | **+2.03** |

No individual task dropped >2 points relative to baseline. See `docs/ablation_table.md` for the full before/after table.

## CLI

| Command | Alias | Description |
|---|---|---|
| `dna-sentinel train` | `dna train` | Train model |
| `dna-sentinel predict` | `dna predict` | Predict on FASTA |
| `dna-sentinel benchmark` | `dna bench` | Benchmark all splits (supports `--ensemble-checkpoint`) |
| `dna-sentinel prepare` | `dna prep` | Build dataset |
| `dna-sentinel prepare-features` | `dna features` | Extract features |
| `dna-sentinel cross-validate` | `dna cv` | k-fold CV |
| `dna-sentinel serve` | `dna serve` | Start API server |
| `dna-sentinel experiment-list` | `dna list` | List experiments |
| — | `dna interpret` | Predict with biological interpretation |

All commands accept short flags (e.g., `-m` for checkpoint, `-f` for FASTA, `-c` for config, `-e` for ensemble checkpoint).

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

## v0.2.0 → v0.3.0 Changes

| Change | Impact |
|---|---|
| `dna` short alias CLI group | Faster command entry, all `--help` complete |
| Short flags (-m, -f, -c, -j, -i, -e) | Reduces typing for common operations |
| Biological interpretation | Confidence labels, CARD matching, disclaimer |
| Web UI (prediction + benchmark) | Interactive offline demo |
| L-BFGS temperature + bias calibration | Task score 84.97% → 85.95% on held-out |
| Per-class weighting + focal loss | Class 1 F1 0.678 → 0.716 (v15) |
| v14+v15 ensemble | Task score → 87.00% |
| Coverage ≥85% | 375 tests, ~89% coverage (stable across runs) |
| Ablation documentation | Before/after table generated |
| Benchmark methodology | Documented splits, protocol, baseline sources |

## Test Coverage

| Module | Coverage |
|---|---:|
| `__init__.py` | 100% |
| `api.py` | 85% |
| `cli.py` | 75% |
| `features.py` | 88% |
| `model.py` | 98% |
| `prepare.py` | 78% |
| `train.py` | 98% (utility functions; training loop excluded as GPU-dependent) |
| `utils.py` | 95% |
| **Total** | **~89%** |

## Intended Use

Cassiopeia Prime is a DNA-only plasmid triage model for education, research prototyping, and comparative benchmarking under a small-data constraint. It flags mobility, AMR cargo, and spread risk signals quickly from sequence alone.

## Limitations

- Evidence windows are model attention scores, not experimentally validated breakpoints.
- The model does **not** run BLAST, gene calling, host metadata lookup, or assembly quality checks.
- RC averaging reduces reverse-complement instability but circular cut-point drift is reduced rather than eliminated.
- 2,048-plasmid limit means the model is inherently data-constrained.
- Mobility class 1 (mobilizable) remains the most challenging class (F1 ≈ 0.72).
- Geographic expansion is a proxy based on ≥15 observed countries; sampling bias affects this.

## Safety

Predictions must not be used as clinical, environmental, regulatory, or biosafety decisions by themselves. Treat outputs as screening signals that need orthogonal validation.
