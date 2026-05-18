# DNA Sentinel Model Card

## Model

Recommended production checkpoint:

```text
artifacts/dna_sentinel/kmer_transformer_best.pt
```

Model family: multi-scale attention-based KmerTransformer.

The KmerTransformer v2 model acts over hashed k-mer features extracted across three windows (512 bp, 2048 bp, and 8192 bp). It combines registered random projection layers with multi-head attention blocks and dynamic evidence pooling to produce highly accurate predictions for mobility, AMR, and plasmid expansion tasks.

## Intended Use

DNA-only screening of plasmid/mobile genetic element sequences for:

- mobility class
- AMR cargo probability
- high-dissemination expansion probability
- ranked sequence windows contributing to risk

The system is for prioritization and research triage. It does not generate DNA, optimize sequences, or provide wet-lab protocols.

## Inputs

Only raw FASTA DNA is accepted at inference. Metadata, taxonomy, geography, host, year, GC/length features, annotations, BLAST hits, and protein embeddings are not model inputs.

## Training Data

Curated local dataset:

| Split | Sequences |
|---|---:|
| Train | 1484 |
| Validation | 299 |
| Test | 265 |

Labels are derived offline from existing project tables. Split construction groups exact duplicates and known backbone groups to reduce leakage.

## Test Metrics

| Task | Metric | Value |
|---|---:|---:|
| Mobility | Accuracy | 0.592 |
| Mobility | Balanced accuracy | 0.594 |
| AMR cargo | AUROC | 0.790 |
| AMR cargo | AUPRC | 0.728 |
| AMR cargo | Brier | 0.189 |
| AMR cargo | ECE | 0.110 |
| Expansion | AUROC | 0.883 |
| Expansion | AUPRC | 0.751 |
| Expansion | Brier | 0.139 |
| Expansion | ECE | 0.090 |

## Stress Metrics

| Check | Value |
|---|---:|
| RC max risk difference | 0.0 |
| RC mean risk difference | 0.0 |
| Approx nearest train-test sketch Jaccard, mean | 0.0032 |
| Approx nearest train-test sketch Jaccard, max | 0.0549 |
| Checkpoint size | 1.40 MB |

## Limitations

- Mobility remains the weakest head under strict group split, though KmerTransformer significantly improved it compared to standard linear models.
- Expansion is highly learnable in this dataset but may reflect historical sampling intensity in the offline label.
- Expansion AUPRC shows a minor drop (-0.038) compared to linear models due to the smoothing effect of multi-scale attention evidence pooling.
- Window explanations are model evidence windows, not validated mechanistic HGT breakpoints.

## Recommended Next Work

- Increase curated dataset to 4096 while preserving group split.
- Add external held-out plasmid collections.
- Validate evidence windows against known mobility/AMR loci without feeding those annotations into the model.
