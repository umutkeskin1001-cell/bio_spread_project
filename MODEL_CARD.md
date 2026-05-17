# DNA Sentinel Model Card

## Model

Recommended production checkpoint:

```text
artifacts/dna_sentinel/kmer.joblib
```

Model family: reverse-complement-consensus hashed k-mer linear classifiers.

The neural sparse-MIL model remains available as a research ablation checkpoint format (`best.pt`) but is not the recommended production model for the current low-data regime.

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
| Mobility | Accuracy | 0.581 |
| Mobility | Balanced accuracy | 0.568 |
| AMR cargo | AUROC | 0.746 |
| AMR cargo | AUPRC | 0.697 |
| AMR cargo | Brier | 0.210 |
| AMR cargo | ECE | 0.210 |
| Expansion | AUROC | 0.867 |
| Expansion | AUPRC | 0.789 |
| Expansion | Brier | 0.145 |
| Expansion | ECE | 0.093 |

## Stress Metrics

| Check | Value |
|---|---:|
| RC max risk difference | 0.0 |
| RC mean risk difference | 0.0 |
| Approx nearest train-test sketch Jaccard, mean | 0.0032 |
| Approx nearest train-test sketch Jaccard, max | 0.0549 |
| Inference latency | 253 ms/sequence |
| Checkpoint size | 2.63 MB |

## Limitations

- Mobility remains the weakest head under strict group split.
- AMR cargo is harder after leakage control than in the initial loose split.
- Expansion is learnable in this dataset but may reflect historical sampling intensity in the offline label.
- Window explanations are model evidence windows, not validated mechanistic HGT breakpoints.

## Recommended Next Work

- Increase curated dataset to 4096 while preserving group split.
- Add external held-out plasmid collections.
- Validate evidence windows against known mobility/AMR loci without feeding those annotations into the model.
