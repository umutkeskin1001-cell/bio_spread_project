# DNA Sentinel Model Card

## Model

Recommended production checkpoint:

```text
artifacts/dna_sentinel/kmer_transformer_best.pt
```

Model family: Genomic Coordinate Gated Bio-Spectral BDSG Transformer.

The DNA Sentinel KmerTransformer integrates a dual-stream architecture with **Genomic Coordinate Aligned Positional Embeddings (GCAPE)** projected via a continuous **Genomic Coordinate MLP (GC-MLP)**. Length-invariant scale coordinates are injected early before **Coordinate-Aware Gated Bilinear Fusion** of lexical and spectral features. In the pooling layer, multi-scale representations are integrated using **Bi-Directional Scale Gating (BDSG)**, forming a soft-differentiable logical AND gate between local motif scales (Scale 0/1) and macro-genomic structures (Scale 2) with zero parameter overhead. The model remains exceptionally lightweight at under 100k parameters and **4.5 ms CPU latency**.

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

## Test Metrics (DNA Sentinel KmerTransformer)

| Task | Metric | Value |
|---|---:|---:|
| Mobility | Accuracy | **0.6830** |
| Mobility | Balanced accuracy | **0.6909** |
| AMR cargo | AUROC | **0.8146** |
| AMR cargo | AUPRC | **0.7697** |
| AMR cargo | Brier | **0.1842** |
| AMR cargo | ECE | **0.1142** |
| Expansion | AUROC | **0.8798** |
| Expansion | AUPRC | **0.8105** |
| Expansion | Brier | **0.1555** |
| Expansion | ECE | **0.1540** |

## Stress Metrics

| Check | Value |
|---|---|
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
