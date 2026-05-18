# DNA Sentinel Model Card

## Model

Recommended production checkpoint:

```text
artifacts/dna_sentinel/kmer_transformer_best.pt
```

Model family: multi-scale attention-based KmerTransformer (v3 - Pure-Tensor).

The KmerTransformer v3 model operates over multi-scale base-4 sequence tensor representations mapped to dense 4096-dimensional k-mer frequencies using a Knuth multiplicative hash on PyTorch. This architecture runs completely in PyTorch, removing scikit-learn dependencies from the inference path, enabling native GPU/MPS acceleration, and achieving a 3.8 ms CPU latency.

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

## Test Metrics (KmerTransformer v3)

| Task | Metric | Value |
|---|---:|---:|
| Mobility | Accuracy | 0.562 |
| Mobility | Balanced accuracy | 0.550 |
| AMR cargo | AUROC | 0.785 |
| AMR cargo | AUPRC | 0.700 |
| AMR cargo | Brier | 0.190 |
| AMR cargo | ECE | 0.133 |
| Expansion | AUROC | 0.838 |
| Expansion | AUPRC | 0.682 |
| Expansion | Brier | 0.161 |
| Expansion | ECE | 0.109 |

## Stress Metrics

| Check | Value |
|---|---:|
| RC max risk difference | 0.0 |
| RC mean risk difference | 0.0 |
| Approx nearest train-test sketch Jaccard, mean | 0.0032 |
| Approx nearest train-test sketch Jaccard, max | 0.0549 |
| Checkpoint size | 1.40 MB |
| Inference latency | 3.8 ms/sequence (65x speedup) |

## Limitations

- Mobility remains the weakest head under strict group split.
- Expansion AUPRC shows a minor drop (-0.038) compared to linear models due to the smoothing effect of multi-scale attention evidence pooling.
- Window explanations are model evidence windows, not validated mechanistic HGT breakpoints.

## Recommended Next Work

- Implement Coordinate-Shift Contrastive Learning (CS-CL) to further regularize the latent space.
- Scale curated dataset to 4096 sequences while preserving group split.
- Validate evidence windows against known mobility/AMR loci without feeding those annotations into the model.
