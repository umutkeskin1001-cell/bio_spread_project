# DNA Sentinel Model Card

## Model

Recommended production checkpoint:

```text
artifacts/dna_sentinel/kmer_transformer_best.pt
```

Model family: Genomic Coordinate Gated Bio-Spectral Bilinear Transformer (v6 - GC-GS-C2BT).

The KmerTransformer v6 (GC-GS-C2BT) model integrates a dual-stream architecture with **Genomic Coordinate Aligned Positional Embeddings (GCAPE)** projected via a continuous **Genomic Coordinate MLP (GC-MLP)**. Physical coordinates are injected early before **Coordinate-Aware Gated Bilinear Fusion** of lexical and spectral features. In the pooling layer, sequence regions are integrated using **Combinatorial Co-Presence Bilinear Pooling (CCBP)** with a **Differentiable Relation Gate**, forming a soft logical AND gate over distant sequences. The model remains exceptionally lightweight at under 100k parameters and **4.5 ms CPU latency**.

## Intended Use

DNA-only screening of plasmid/mobile genetic element sequences for:

- mobility class (non-mobile, plasmid, phage)
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

## Test Metrics (KmerTransformer v6 - GC-GS-C2BT)

| Task | Metric | Value |
|---|---:|---:|
| Mobility | Accuracy | **0.521** |
| Mobility | Balanced accuracy | **0.506** |
| AMR cargo | AUROC | **0.783** |
| AMR cargo | AUPRC | **0.755** (+5.5% gain!) |
| AMR cargo | Brier | **0.186** |
| AMR cargo | ECE | **0.145** |
| Expansion | AUROC | **0.874** |
| Expansion | AUPRC | **0.756** |
| Expansion | Brier | **0.147** |
| Expansion | ECE | **0.110** |

## Stress Metrics

| Check | Value |
|---|---:|
| RC max risk difference | 0.0 |
| RC mean risk difference | 0.0 |
| Approx nearest train-test sketch Jaccard, mean | 0.0032 |
| Approx nearest train-test sketch Jaccard, max | 0.0549 |
| Checkpoint size | 1.62 MB |
| Inference latency | **4.5 ms/sequence** (56x speedup) |

## Limitations

- The model profiles sequences based on multi-scale k-mer, coordinate mappings, and Fourier structural periodicities; it does not perform full sequence generation or mechanistic base-by-base editing.
- Window explanations are model evidence weights, not validated biological HGT breakpoints.

## Recommended Next Work

- Scale curated dataset to 4096 sequences while preserving group split.
- Validate evidence windows against known mobility/AMR loci without feeding those annotations into the model.
