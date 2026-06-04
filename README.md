# Cassiopeia Prime

**Cassiopeia Prime** is a compact, DNA-only plasmid risk model. It predicts mobility class, antimicrobial resistance (AMR) probability, and geographic expansion risk directly from FASTA sequence — without BLAST, metadata, or annotation pipelines.

**Training cap:** 2,048 plasmids maximum (1,466 train / 286 val / 296 test / 1,200 held-out).

## Benchmark (Held-out)

| Metric | Value | 95% CI |
|---:|---:|---:|
| Task Score | **85.95%** | 85.18–86.75% |
| Mobility BA | 76.92% | 74.63–79.31% |
| AMR AUROC | 93.91% | 92.54–95.15% |
| Expansion AUROC | 87.01% | 85.02–88.93% |

Parameters: 568,437 | Checkpoint: 5.81 MB | 371 tests, 89.47% coverage

## Quick Start

```bash
pip install -e ".[dev]"

# Short aliases available:
dna prep -c config/cassiopeia_prime.yaml      # build dataset
dna features -c config/cassiopeia_prime.yaml   # extract features
dna train -c config/cassiopeia_prime.yaml      # train model
dna bench -m artifacts/.../cassiopeia_best.pt  # benchmark
dna predict -m artifacts/.../cassiopeia_best.pt -f query.fa --interpret  # predict
dna serve -m artifacts/.../cassiopeia_best.pt  # start API
```

## CLI Reference

| Command | Alias | Short flags | Description |
|---|---|---|---|
| `dna-sentinel train` | `dna train` | `-c` | Train model |
| `dna-sentinel predict` | `dna predict` | `-m`, `-f`, `-j`, `-i` | Predict on FASTA |
| `dna-sentinel benchmark` | `dna bench` | `-m`, `-d`, `-o` | Benchmark all splits |
| `dna-sentinel prepare` | `dna prep` | `-c` | Build dataset from FASTA+labels |
| `dna-sentinel prepare-features` | `dna features` | `-c` | Extract k-mer+structural features |
| `dna-sentinel cross-validate` | `dna cv` | `-c`, `-k` | k-fold cross-validation |
| `dna-sentinel serve` | `dna serve` | `-m`, `-p` | Start FastAPI server |
| `dna-sentinel experiment-list` | `dna list` | — | List experiment runs |
| — | `dna interpret` | `-m`, `-f` | Predict + biological interpretation |

## Biological Interpretation

Predictions include **confidence labels** (HIGH ≥ 0.80, MEDIUM 0.60–0.79, LOW < 0.60):
- **Mobility**: Non-mobilizable / Mobilizable / Conjugative + biological description
- **AMR**: CARD family matching at HIGH confidence
- **Expansion**: Synthetic inference from mobility+AMR co-occurrence
- **Disclaimer**: "Tarama sinyalidir; klinik, çevresel veya biyogüvenlik kararlarında tek başına kullanılamaz."

## Web Interface

Static pages served from `web/` — fully offline:

```bash
python3 -m http.server 4173 --directory web
```

- **Predict** (`index.html`): FASTA/text input → scores, confidence, evidence windows, interpretation
- **Benchmark** (`benchmark.html`): Cassiopeia vs DNABERT, DNABERT-2, Nucleotide Transformer, PLSDB baseline

## Project Structure

```
config/cassiopeia_prime.yaml   # champion config (model + training + features)
src/dna_sentinel/
  model.py                     # Cassiopeia model (encoder, heads, evidence pool)
  features.py                  # canonical k-mer extraction, structural features
  train.py                     # training loop, calibration, cross-validation
  cli.py                       # Click CLI + `dna` short alias group
  utils.py                     # prediction, interpretation, metrics, experiment tracking
  api.py                       # FastAPI server
  prepare.py                   # dataset preparation, split generation
web/                           # static HTML/CSS/JS demo
tests/                         # 371 tests, 89.47% coverage
docs/                          # benchmark.json, model card, TÜBİTAK report
experiments/                   # timestamped experiment logs
```

## Development

```bash
make install     # pip install -e ".[dev]"
make test        # pytest with coverage
make lint        # ruff check
```

## Calibration (ECE)

| Split | Mobility ECE | AMR ECE | Expansion ECE |
|---|---:|---:|---:|
| Validation | 0.085 | 0.040 | 0.084 |
| Test | 0.112 | 0.036 | 0.069 |
| Held-out | 0.116 | 0.064 | 0.044 |

L-BFGS temperature + bias scaling on validation logits. All tasks ECE < 0.12.

## Ablation (before → after)

| Metric | Before | After | Δ |
|---|---:|---:|---:|
| Held-out Task Score | 84.97% | **85.95%** | +0.98 |
| Mobility BA | 76.75% | 76.92% | +0.17 |
| AMR AUROC | 93.47% | 93.91% | +0.44 |
| Expansion AUROC | 84.68% | 87.01% | +2.33 |

No task dropped >2 points. Improvements: focal loss γ=0.5, consistency weight tuning, L-BFGS calibration.

## Limitations

- Evidence windows are attention scores, not validated breakpoints
- No BLAST, gene calling, host metadata, or assembly quality checks
- RC averaging reduces but does not eliminate circular cut-point drift
- 2,048-plasmid cap → data constrained
- Mobility class 1 (mobilizable) remains challenging (F1 ≈ 0.68)
- Expansion is a proxy based on ≥15 observed countries

## License

MIT — Umut Keskin 2025
