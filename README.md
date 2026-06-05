# Cassiopeia Prime

**Cassiopeia Prime** is a compact, DNA-only plasmid risk model. It predicts mobility class, antimicrobial resistance (AMR) probability, and geographic expansion risk directly from FASTA sequence — without BLAST, metadata, or annotation pipelines.

**Training cap:** 2,048 plasmids maximum (1,466 train / 286 val / 296 test / 1,200 held-out).
**Champion:** v14+v15 ensemble (`artifacts/cassiopeia_prime_v15/cassiopeia_best.pt` + `artifacts/cassiopeia_prime_v14/cassiopeia_best.pt`), weights 0.47/0.53.

## Benchmark (Held-out)

| Metric | Value | 95% CI |
|---:|---:|---:|
| Task Score | **87.00%** | 86.20–87.78% |
| Mobility BA | 78.33% | 75.92–80.67% |
| AMR AUROC | 93.89% | 92.58–95.04% |
| Expansion AUROC | 88.78% | 87.00–90.55% |
| **Class 1 (mobilizable) F1** | **0.725** | — |

Parameters: 568,437 | Checkpoint: 5.81 MB × 2 | 375 tests, ~89% coverage

> **Inference mode:** Numbers above are produced by `dna-sentinel benchmark --ensemble-checkpoint`
> (default), which evaluates pre-cached features without reverse-complement averaging. This matches
> the protocol used during model selection and is fully reproducible from the committed
> feature cache. Production-style reverse-complement-averaged numbers (matching the
> inference path used by `dna predict` / `dna interpret`) are ~1–2 points higher on the
> held-out split. Regenerate with `dna-sentinel benchmark --rc-average --ensemble-checkpoint ...`.
> Reports include an `inference_mode` field (`cached_features` or `rc_averaged`).

## Quick Start

```bash
pip install -e ".[dev]"

# Short aliases available:
dna prep -c config/cassiopeia_prime.yaml      # build dataset
dna features -c config/cassiopeia_prime.yaml   # extract features
dna train -c config/cassiopeia_prime.yaml      # train model
dna bench -m artifacts/.../cassiopeia_best.pt -e artifacts/.../cassiopeia_best.pt --ensemble-weight 0.53  # benchmark ensemble
dna predict -m artifacts/.../cassiopeia_best.pt -f query.fa --interpret  # predict
dna serve -m artifacts/.../cassiopeia_best.pt  # start API
```

## CLI Reference

| Command | Alias | Short flags | Description |
|---|---|---|---|
| `dna-sentinel train` | `dna train` | `-c` | Train model |
| `dna-sentinel predict` | `dna predict` | `-m`, `-f`, `-j`, `-i` | Predict on FASTA |
| `dna-sentinel benchmark` | `dna bench` | `-m`, `-d`, `-o`, `-e` | Benchmark all splits (supports ensemble) |
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

Static pages served from `web/` — fully offline, no backend required:

```bash
python3 -m http.server 4173 --directory web
# then open http://localhost:4173/index.html
```

The browser demo runs a lightweight k-mer + motif heuristic and is **not** the
trained model. Numbers on the **Benchmark** page come from a frozen snapshot
of the champion ensemble (`cassiopeia_prime_v15` + `cassiopeia_prime_v14`).

- **Predict** (`index.html`): FASTA/text input → mobility class + probabilities,
  AMR/expansion percentages, evidence windows, biological interpretation,
  class-probability bars, and a copyable reproduction command for the CLI.
- **Benchmark** (`benchmark.html`): Cassiopeia vs DNABERT, DNABERT-2, Nucleotide
  Transformer and a k-mer logistic-regression baseline, with split-wise
  performance, key metrics, methodology and limitations tabs.

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
tests/                         # 375 tests, ~89% coverage
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
|---:|---:|---:|---:|
| Validation | 0.148 | 0.061 | 0.055 |
| Test | 0.135 | 0.055 | 0.052 |
| Held-out | 0.164 | 0.051 | 0.038 |

L-BFGS temperature + bias scaling on validation logits (per model before ensemble averaging).

## Ablation (before → after)

| Metric | Before | After (ensemble) | Δ |
|---|---|---:|---:|---:|
| Held-out Task Score | 84.97% | **87.00%** | +2.03 |
| Mobility BA | 76.75% | 78.33% | +1.58 |
| AMR AUROC | 93.47% | 93.89% | +0.42 |
| Expansion AUROC | 84.68% | 88.78% | +4.10 |
| **Class 1 F1** | — | **0.725** | — |

Improvements: L-BFGS calibration, consistency weight tuning, per-class mobility weighting + focal γ=0.5 (v15), v14+v15 ensemble.

## Limitations

- Evidence windows are attention scores, not validated breakpoints
- No BLAST, gene calling, host metadata, or assembly quality checks
- RC averaging reduces but does not eliminate circular cut-point drift
- 2,048-plasmid cap → data constrained
- Mobility class 1 (mobilizable) remains challenging (F1 ≈ 0.72)
- Expansion is a proxy based on ≥15 observed countries

## License

MIT — Umut Keskin 2025
