# Cassiopeia Model Card

## Model Overview

| Property | Value |
|----------|-------|
| **Name** | Cassiopeia |
| **Type** | Multi-task neural network (FRP + F-LoRA + GLUMixer + evidence pooling) |
| **Parameters** | 418,000 |
| **Input** | Raw DNA sequence (FASTA) → multi-scale canonical k-mer features (28 windows × 2728-dim) |
| **Outputs** | Mobility class (3-class), AMR probability, expansion probability, per-window evidence scores |
| **Inference latency** | ~4.5 ms/sequence on CPU |
| **Checkpoint size** | ~1.4 MB (FRP matrix regenerated from seed, not stored) |

## Architecture Components

1. **FRP** — Fast Random Projection (2728 → 256), seeded deterministic ternary matrix (1/6 each ±1, 2/3 zero)
2. **F-LoRA** — Low-rank correction (rank 8) adapts the FRP projection
3. **Bottleneck** — LayerNorm + Linear(256→128) + GELU
4. **ContextGate** — Gated cross-window context modulation via learned gate
5. **2× GLUMixer** — Hybrid token-mixing (window axis) + channel-mixing (feature axis) with GELU gating
6. **Stochastic Depth** — DropPath regularization (rate 0.1)
7. **Task Adapters** — Bottleneck adapters (128→8→128) per task, applied to shared representation
8. **Multi-Query Evidence Pooling** — Attention-weighted pooling over windows (1 head)
9. **Deep Supervision** — Auxiliary classification heads on intermediate layer outputs (weight 0.3)
10. **Task heads** — Mobility: Linear(128→3), AMR: Linear(128→1), Expansion: Linear(128+3+1→1)

## Training Data

Curated from PLSDB plasmid database. Split is group-aware (exact duplicates and known backbone groups are kept together).

| Split | Sequences |
|-------|----------:|
| Train | 1,414 |
| Validation | 280 |
| Test | 354 |
| Held-out test | 1,200 |

Features: multi-scale canonical k-mer counts at window sizes 512 bp, 2048 bp, 8192 bp (k=4–6).

## Performance (Held-out Test)

| Task | Metric | Value |
|------|--------|------:|
| Mobility | Balanced accuracy | **77.94%** |
| AMR | AUROC | **92.51%** |
| Expansion | AUROC | **80.36%** |

## Training Configuration

- Optimizer: AdamW (backbone lr=3e-4, head lr=3e-4, weight decay=0.05)
- Scheduler: Linear warmup (5 epochs) → Cosine decay to 1e-5
- Loss: Focal loss (γ=2.0) for binary tasks, cross-entropy (label smoothing 0.1) for mobility
- Uncertainty weighting: Learned log-variance per task
- Regularization: Stochastic depth (0.1), manifold mixup (α=0.3), WindowDropout (0.15)
- Deep supervision aux loss weight: 0.3
- Gradient accumulation: 2 steps
- Early stopping: 25 epochs patience

## Limitations

- The model profiles sequences based on multi-scale k-mer histograms with random projection, not base-level editing
- Window evidence scores indicate model attention, not validated HGT breakpoints
- Performance on out-of-distribution sequences (non-plasmid, novel backbones) may degrade
