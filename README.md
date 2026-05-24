# Cassiopeia

**Cassiopeia** is a lightweight, DNA-only plasmid risk prediction model. It jointly predicts mobility class (conjugative / mobilizable / non-mobilizable), antimicrobial resistance (AMR) cargo probability, and host-range expansion risk directly from raw nucleotide sequence — no BLAST, no metadata, no annotation pipeline required.

## Architecture

```
Input (k-mer features: 28 windows × 2728-dim)
  → FRP (Fast Random Projection: 2728 → 256)
  → F-LoRA correction (rank 8)
  → Bottleneck (256 → 128)
  → ContextGate (cross-window context modulation)
  → 2× GLUMixer (token & channel mixing)
  → Task-conditioned bottleneck adapters
  → Multi-Query Evidence Pooling
  → Task heads (mobility 3-class, AMR binary, expansion binary)
```

- **418K parameters**, ~4.5 ms/seq CPU inference
- Deterministic FRP matrix (regenerated from seed at load — zero storage)
- Reverse-complement consistent by construction

## Quick Start

```bash
# Install
pip install -e .

# Full pipeline
dna-sentinel prepare --config config/dna_sentinel.yaml
dna-sentinel prepare-features --config config/dna_sentinel.yaml
dna-sentinel train --config config/dna_sentinel.yaml

# Predict on FASTA
dna-sentinel predict --checkpoint artifacts/dna_sentinel/cassiopeia_best.pt --fasta query.fasta

# Serve API
dna-sentinel serve --checkpoint artifacts/dna_sentinel/cassiopeia_best.pt --port 8000
```

## Project Structure

```
config/
  dna_sentinel.yaml          # Single unified config
src/dna_sentinel/
  model.py                   # Cassiopeia model + CassiopeiaConfig
  train.py                   # Training loop, evaluation, calibration
  features.py                # CanonicalKmerExtractor (multi-scale)
  prepare.py                 # Dataset preparation + group-aware split
  utils.py                   # I/O, metrics, WindowDropout, InferenceService
  cli.py                     # CLI entrypoints
  api.py                     # FastAPI server
tests/                       # 47 unit tests
```

## Tests

```bash
pytest tests/ -v
```

## License

MIT
