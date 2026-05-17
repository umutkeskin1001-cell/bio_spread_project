# DNA Sentinel Benchmark Report

## Summary

The original metadata-heavy BioSpread direction was replaced with a DNA-only system. The first neural sparse-MIL model was scientifically attractive but underperformed a simpler hashed k-mer model in the low-data regime. The final production recommendation is therefore the k-mer model, with the neural model retained as a research ablation path.

This was not chosen because it is simpler; it was chosen because the benchmark made it the stronger, more sample-efficient model.

## Dataset

Prepared with:

```bash
dna-sentinel prepare --config config/dna_sentinel.yaml
```

Current dataset:

| Split | N | Mobility distribution | AMR distribution | Expansion distribution |
|---|---:|---|---|---|
| Train | 1484 | 0:574, 1:396, 2:514 | 0:767, 1:717 | 0:796, 1:688 |
| Validation | 299 | 0:137, 1:76, 2:86 | 0:159, 1:140 | 0:192, 1:107 |
| Test | 265 | 0:91, 1:75, 2:99 | 0:154, 1:111 | 0:177, 1:88 |

Split is group-aware and exact-duplicate-aware. Metadata is used only to construct labels and split groups, not as model input.

## Models Tested

| Model | Outcome |
|---|---|
| Neural sparse-MIL DNA model | Good concept, weaker metrics under low data |
| Hashed k-mer linear model | Best sample-efficient production model |
| RC-consensus k-mer model | Final model; exact reverse-complement prediction consistency |

## Final Test Metrics

| Task | Metric | Final k-mer |
|---|---:|---:|
| Mobility | Accuracy | 0.5811 |
| Mobility | Balanced accuracy | 0.5682 |
| AMR cargo | AUROC | 0.7463 |
| AMR cargo | AUPRC | 0.6970 |
| AMR cargo | Brier | 0.2104 |
| AMR cargo | ECE | 0.2096 |
| Expansion | AUROC | 0.8674 |
| Expansion | AUPRC | 0.7886 |
| Expansion | Brier | 0.1449 |
| Expansion | ECE | 0.0934 |

## Stress Tests

| Test | Result | Interpretation |
|---|---:|---|
| Reverse-complement max risk diff | 0.0 | Passed |
| Reverse-complement mean risk diff | 0.0 | Passed |
| Approx nearest train-test sketch Jaccard mean | 0.0032 | Low similarity |
| Approx nearest train-test sketch Jaccard max | 0.0549 | No obvious near-duplicate in sampled audit |
| Inference latency | 253 ms/sequence | Acceptable CPU path |
| Checkpoint size | 2.63 MB | Production-friendly |

## Ruthless Assessment

Strong:

- The system is truly DNA-only at inference.
- Expansion prediction is strong after group-aware leakage control.
- Reverse-complement consistency is exact for risk predictions.
- Disk footprint and model size are small.

Weak:

- Mobility classification is not yet publication-grade.
- AMR dropped when moving from a loose split to a strict group split.
- Calibration is acceptable for expansion, weak for AMR.
- Evidence windows need biological validation.

Decision:

The project is production-clean and scientifically honest, but the current model is not yet a complete high-impact paper by metrics alone. It is a strong foundation and a defensible TÜBİTAK/ERC prototype. The next decisive experiment is a 4096-sequence strict split plus external holdout.
