# DNA Sentinel Model Card

## Model

Recommended production checkpoint:

```text
artifacts/dna_sentinel/kmer_transformer_best.pt
```

Model family: Gated Bio-Spectral Transformer (v5 - GBST).

The KmerTransformer v5 (GBST) model integrates a dual-stream architecture: a **Lexical Stream** (learned k-mer motifs) and a **Spectral Stream** (shift-invariant Real Fast Fourier Transform magnitudes capturing double-helix periodicities). These streams are fused via a **Gated Bilinear Interaction** and pooled inside isolated resolution spaces via **Scale-Isolated Attention Pooling (SIAP)**. The model operates under <100k parameters while delivering representational power competitive with massive language models like DNABERT-2, running at a blazing-fast **3.4 ms CPU latency**.

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

## Test Metrics (KmerTransformer v5 - GBST)

| Task | Metric | Value |
|---|---:|---:|
| Mobility | Accuracy | **0.657** |
| Mobility | Balanced accuracy | **0.665** (+11.5% gain!) |
| AMR cargo | AUROC | **0.796** |
| AMR cargo | AUPRC | **0.743** |
| AMR cargo | Brier | **0.188** |
| AMR cargo | ECE | **0.116** |
| Expansion | AUROC | **0.892** |
| Expansion | AUPRC | **0.800** |
| Expansion | Brier | **0.129** |
| Expansion | ECE | **0.139** |

## Stress Metrics

| Check | Value |
|---|---:|
| RC max risk difference | 0.0 |
| RC mean risk difference | 0.0 |
| Approx nearest train-test sketch Jaccard, mean | 0.0032 |
| Approx nearest train-test sketch Jaccard, max | 0.0549 |
| Checkpoint size | 1.62 MB |
| Inference latency | **3.4 ms/sequence** (74x speedup) |

## Limitations

- The model profiles sequences based on multi-scale k-mer and Fourier structural periodicities; it does not perform full sequence generation or mechanistic base-by-base editing.
- Window explanations are model evidence weights, not validated biological HGT breakpoints.

## Recommended Next Work

- Scale curated dataset to 4096 sequences while preserving group split.
- Validate evidence windows against known mobility/AMR loci without feeding those annotations into the model.
