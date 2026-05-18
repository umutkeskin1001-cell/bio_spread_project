# DNA Sentinel

DNA Sentinel is a small, compute-efficient, sequence-only model for mobile genetic element risk analysis. It predicts mobility, AMR cargo, expansion risk, and sparse evidence windows from raw DNA sequences only.

The project intentionally rejects metadata-heavy shortcuts. At inference time the model accepts only FASTA DNA. No taxonomy, host, geography, temporal context, GC/length feature, annotation, BLAST hit, or protein embedding is passed to the model.

## Scientific Aim

The central question is whether plasmid-level mobile AMR risk can be inferred from nucleotide organization alone under strict low-data and low-compute constraints.

DNA Sentinel v3 uses a multi-scale, pure-tensor **KmerTransformer** architecture:

```text
FASTA
  -> multi-scale sequence windows (512, 2048, 8192 bp)
  -> base-4 nucleotide integer conversion (PyTorch tensor)
  -> pure-tensor unfold & multiplicative hashing (CUDA/MPS-native)
  -> multi-head attention Transformer encoder
  -> sparse evidence pooling with entropy regularization
  -> multi-task prediction heads & temperature scaling
```

This design guarantees exact reverse-complement consistency, low storage footprint, lightning-fast inference speed (3.8 ms/sequence), and attention-based evidence window interpretability, completely bypassing string manipulations.

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

To prepare sequence jsonl splits and label tensors:

```bash
# Prepare raw data splits:
dna-sentinel prepare --config config/dna_sentinel.yaml

# Preprocess multi-scale k-mer features:
dna-sentinel prepare-kmer-transformer --config config/dna_sentinel.yaml
```

Outputs:

```text
data/dna_sentinel/train.jsonl
data/dna_sentinel/train_features.pt
data/dna_sentinel/train_labels.pt
data/dna_sentinel/val_features.pt
data/dna_sentinel/val_labels.pt
data/dna_sentinel/test_features.pt
data/dna_sentinel/test_labels.pt
```

## Train

Recommended low-data production model (KmerTransformer v3):

```bash
dna-sentinel train-kmer-transformer --config config/dna_sentinel.yaml
```

Checkpoints:

```text
artifacts/dna_sentinel/kmer_transformer_best.pt
artifacts/dna_sentinel/kmer_transformer_history.json
```

## Evaluate

```bash
dna-sentinel evaluate-kmer-transformer \
  --checkpoint artifacts/dna_sentinel/kmer_transformer_best.pt \
  --data-dir data/dna_sentinel
```

## Predict / Serve

The model can be queried via the CLI or served as a lightweight FastAPI service:

```bash
# Serve the API:
uvicorn dna_sentinel.api:app --host 0.0.0.0 --port 8000
```

To run inside Docker:

```bash
docker build -t dna-sentinel .
docker run --rm -p 8000:8000 \
  -e DNA_SENTINEL_CHECKPOINT=artifacts/dna_sentinel/kmer_transformer_best.pt \
  dna-sentinel
```

Example `/predict` JSON response format:

```json
{
  "sequence_id": "query",
  "mobility_probs": [0.2, 0.3, 0.5],
  "amr_probability": 0.79,
  "expansion_probability": 0.75,
  "risk_score": 0.49,
  "top_windows": [{"start": 256.0, "end": 768.0, "weight": 0.18}]
}
```

## Current Honest Benchmark

Current strict group-aware split, `2048` curated sequences:

| Task | Metric | Baseline k-mer | KmerTransformer (v2) | KmerTransformer (v3) |
|---|---:|---:|---:|---:|
| Mobility | Accuracy | 0.581 | **0.592** | 0.562 |
| Mobility | Balanced accuracy | 0.568 | **0.594** | 0.550 |
| AMR cargo | AUROC | 0.746 | **0.790** | 0.785 |
| AMR cargo | AUPRC | 0.697 | **0.728** | 0.700 |
| Expansion | AUROC | 0.867 | **0.883** | 0.838 |
| Expansion | AUPRC | 0.789 | **0.751** | 0.682 |

Stress checks:

| Check | Baseline k-mer | KmerTransformer (v2) | KmerTransformer (v3) |
|---|---:|---:|---:|
| Reverse-complement max risk difference | 0.0 | 0.0 | **0.0** |
| Approx nearest train-test sketch Jaccard max | 0.0549 | 0.0549 | **0.0549** |
| Checkpoint size | 2.63 MB | 1.40 MB | **1.40 MB** |
| Inference latency | 253 ms/sequence | 24.7 ms/sequence | **3.8 ms/sequence** |

## Safety Boundary

DNA Sentinel is an analysis and prioritization tool. It does not generate DNA, optimize sequences, propose edits, or provide wet-lab protocols.
