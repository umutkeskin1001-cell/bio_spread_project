# DNA Sentinel

DNA Sentinel is a small, compute-efficient, sequence-only model for mobile genetic element risk analysis. It predicts mobility, AMR cargo, expansion risk, and sparse evidence windows from raw DNA sequences only.

The project intentionally rejects metadata-heavy shortcuts. At inference time the model accepts only FASTA DNA. No taxonomy, host, country, year, GC/length feature, annotation, BLAST hit, or protein embedding is passed to the model.

## Scientific Aim

The central question is whether plasmid-level mobile AMR risk can be inferred from nucleotide organization alone under strict low-data and low-compute constraints.

DNA Sentinel uses a compact reverse-complement-consistent sparse multiple-instance architecture:

```text
FASTA
  -> DNA canonicalization
  -> overlapping sequence windows
  -> reverse-complement shared convolutional encoder
  -> sparse evidence pooling
  -> calibrated task heads
  -> top risk windows
```

This design favors biological invariance, leakage control, and interpretable local evidence over brute-force pretraining.

## What The Model Uses

Allowed model input:

- Raw DNA strings
- Tokenized A/C/G/T/N ids derived directly from DNA

Forbidden model input:

- Taxonomy
- Host or geography
- Year or temporal history
- GC content, sequence length, replicon counts, mobility annotations, AMR hit counts
- Protein embeddings or translated ORFs
- BLAST/search outputs

Metadata tables are used only offline to derive training labels and split audits. They are not included in inference artifacts.

## Tasks

| Head | Output | Label source used offline |
|---|---:|---|
| Mobility | non-mobilizable / mobilizable / conjugative | prepared plasmid mobility labels |
| AMR cargo | probability | AMR consensus table |
| Expansion risk | probability | high-dissemination backbone label derived offline from country breadth |
| Evidence windows | ranked windows | weakly supervised sparse evidence |

The final `risk_score` is the geometric mean of mobile probability, AMR probability, and expansion probability.

## Install

```bash
python3 -m pip install -e ".[dev]"
```

## Prepare A Small Curated Dataset

The default config builds a 512-sequence dataset from local project inputs:

```bash
dna-sentinel prepare --config config/dna_sentinel.yaml
```

Outputs:

```text
data/dna_sentinel/train.jsonl
data/dna_sentinel/val.jsonl
data/dna_sentinel/test.jsonl
data/dna_sentinel/split.json
```

The split is cluster-aware to reduce near-duplicate leakage.

## Train

Recommended low-data production model:

```bash
dna-sentinel train-kmer --config config/dna_sentinel.yaml
```

Neural sparse-MIL research model:

```bash
dna-sentinel train --config config/dna_sentinel.yaml
```

Checkpoint:

```text
artifacts/dna_sentinel/kmer.joblib
artifacts/dna_sentinel/best.pt
artifacts/dna_sentinel/history.json
```

## Evaluate

```bash
dna-sentinel evaluate \
  --checkpoint artifacts/dna_sentinel/best.pt \
  --data-dir data/dna_sentinel
```

For the recommended k-mer model:

```bash
dna-sentinel evaluate-kmer \
  --checkpoint artifacts/dna_sentinel/kmer.joblib \
  --data-dir data/dna_sentinel
```

Reported metrics include AUROC, AUPRC, Brier score, ECE, accuracy, and balanced accuracy.

## Predict

```bash
dna-sentinel predict \
  --checkpoint artifacts/dna_sentinel/best.pt \
  --fasta query.fa \
  --json
```

For the recommended k-mer model:

```bash
dna-sentinel predict-kmer \
  --checkpoint artifacts/dna_sentinel/kmer.joblib \
  --fasta query.fa \
  --json
```

Example JSON fields:

```json
{
  "sequence_id": "query",
  "mobility_probs": [0.2, 0.3, 0.5],
  "amr_probability": 0.71,
  "expansion_probability": 0.64,
  "risk_score": 0.61,
  "top_windows": [{"start": 0, "end": 1024, "weight": 0.22}]
}
```

## Docker

```bash
docker build -t dna-sentinel .
docker run --rm -p 8000:8000 \
  -e DNA_SENTINEL_CHECKPOINT=artifacts/dna_sentinel/best.pt \
  dna-sentinel
```

## Benchmark Plan

Required comparisons:

- Majority baseline
- Logistic regression over hashed k-mer counts
- Small CNN with mean pooling
- DNA Sentinel without reverse-complement consensus
- DNA Sentinel without sparse evidence pooling
- DNA Sentinel full model

Required stress tests:

- Reverse-complement invariance
- Near-duplicate split leakage audit
- Short-fragment degradation
- Long-plasmid window budget
- Train-size sweep: 64, 128, 512, 4096

## Current Honest Benchmark

Current strict group-aware split, `2048` curated sequences:

| Task | Metric | Test |
|---|---:|---:|
| Mobility | Accuracy | 0.581 |
| Mobility | Balanced accuracy | 0.568 |
| AMR cargo | AUROC | 0.746 |
| AMR cargo | AUPRC | 0.697 |
| Expansion | AUROC | 0.867 |
| Expansion | AUPRC | 0.789 |

Stress checks:

| Check | Value |
|---|---:|
| Reverse-complement max risk difference | 0.0 |
| Approx nearest train-test sketch Jaccard max | 0.0549 |
| Checkpoint size | 2.63 MB |
| Inference latency | 253 ms/sequence |

Ruthless assessment: expansion prediction is strong enough for a serious prototype; AMR and mobility need external validation and more data before making a high-impact publication claim.

## Safety Boundary

DNA Sentinel is an analysis and prioritization tool. It does not generate DNA, optimize sequences, propose edits, or provide wet-lab protocols.
