# Cassiopeia

**Cassiopeia Prime** is a compact, DNA-only plasmid risk model. It predicts mobility class, antimicrobial resistance cargo probability, and geographic expansion risk directly from FASTA sequence, without BLAST, metadata, or an annotation pipeline. Deployment uses reverse-complement averaged inference for stronger real-world robustness.

The project is optimized for a hard training budget of **2,048 plasmids maximum**. Current local data uses 1,414 train, 280 validation, 354 test, and 1,200 held-out plasmids.

## Current Champion

```
Input: 56 multi-scale windows x 2,728 canonical k-mer features
  -> deterministic FRP 2728 -> 320
  -> F-LoRA rank 12
  -> structural feature fusion
  -> circular plasmid positional encoding
  -> lightweight window motif convolution
  -> 3 x GLUMixer
  -> task adapters + task-specific evidence pooling
  -> calibrated mobility / AMR / expansion heads
  -> reverse-complement averaged deployment
```

| Property | Value |
|---:|---:|
| Trainable parameters | 580,694 |
| Checkpoint size | 5.59 MB |
| Cached CPU latency | 0.66 ms/sequence |
| FASTA CPU latency, 6 kb query | 24.16 ms/sequence |
| Held-out task score | 85.49% |

Held-out audit for `artifacts/cassiopeia_prime/cassiopeia_best.pt`:

| Task | Metric | Value |
|---:|---:|---:|
| Mobility | Balanced accuracy | 79.90% |
| AMR | AUROC | 93.28% |
| Expansion | AUROC | 83.30% |

## Quick Start

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .

# Rebuild the 2,048-plasmid training set and 56-window features.
dna-sentinel prepare --config config/cassiopeia_prime.yaml
dna-sentinel prepare-features --config config/cassiopeia_prime.yaml

# Train and benchmark the champion profile.
dna-sentinel train --config config/cassiopeia_prime.yaml
dna-sentinel benchmark \
  --checkpoint artifacts/cassiopeia_prime/cassiopeia_best.pt \
  --data-dir data/dna_sentinel

# Predict on FASTA.
dna-sentinel predict \
  --checkpoint artifacts/cassiopeia_prime/cassiopeia_best.pt \
  --fasta data/dna_sentinel/query.fa \
  --json
```

## Static Demo

The GitHub Pages site is pure static HTML/CSS/JS:

```bash
python3 -m http.server 4173 --directory docs
```

Then open `http://127.0.0.1:4173`.

## Development

```bash
python3 -m pytest -q
python3 -m ruff check src tests
```

`requirements_inference.txt` intentionally excludes scikit-learn; metric code imports it lazily so the inference/API path stays lighter.

## Project Structure

```
config/cassiopeia_prime.yaml  # champion training + feature config
src/dna_sentinel/model.py     # Cassiopeia model and checkpoint loading
src/dna_sentinel/features.py  # canonical k-mer and structural features
src/dna_sentinel/train.py     # training, evaluation, calibration
src/dna_sentinel/cli.py       # prepare/train/evaluate/benchmark/predict/serve
docs/                         # GitHub Pages static demo and reports
tests/                        # unit and integration tests
```

## License

MIT
