# Cold-Start Architecture Fix Task

## Context
BioSpread model predicts bacterial host range from genomic sequences. Two prediction modes:
- **Temporal**: uses time-series snapshots (sequence features) + static (taxonomy, categorical) → AUC ~0.85
- **Cold-start**: static features ONLY (temporal features masked) → need to improve AUC

## The Problem
The model has a `ColdStartEncoder` module and feature flags (`use_evidential`, `use_cagrad`, `use_retrieval`). When ALL flags are `false`, cold-start AUC = **0.694** (uses old gate→hazard_head path, ColdStartEncoder exists but not used for evaluation). When any flag is `true` AND cold-start evaluation routes through `cold_logits`, AUC drops to **~0.37** (worse than random).

I.e., the `ColdStartEncoder` produces garbage predictions even though it's trained with BCE loss (`lambda_cold=0.5`) on all training samples.

## What Needs to Be Done
1. Diagnose why `ColdStartEncoder` fails when feature flags are enabled
2. Fix the architecture so cold-start AUC ≥ 0.694 with all flags enabled
3. Keep backward compatibility (flags default to `false`)
4. All 43 existing tests must pass

## Key Files
- `src/bio_spread/models/sovereign.py` — model forward pass, cold path routing
- `src/bio_spread/models/trainer.py` — loss computation, CAGrad, cold calibration & evaluation
- `src/bio_spread/models/components.py` — ColdStartEncoder, CAGradProjector, EvidentialHazardHead, UncertaintyProtoRetriever
- `config/default.yaml` — feature flags and hyperparams
- `tests/test_redesign.py` — 43 tests

## Commands
```bash
# Tests (ignore API/serving — they need infrastructure):
python3 -m pytest tests/ -v --ignore=tests/test_api.py --ignore=tests/test_serving.py

# Training (~15 min):
python3 -m bio_spread.cli.main train --config config/default.yaml --feature-dir data/features

# Evaluation (~3 min):
python3 -m bio_spread.cli.main evaluate --model-path artifacts/<run>/best_model.pt --config config/default.yaml --feature-dir data/features
```

## Success Criteria
- All tests pass
- Cold-start AUC ≥ 0.694 with flags ON
- Temporal AUC doesn't regress significantly (±0.01 of ~0.85)
