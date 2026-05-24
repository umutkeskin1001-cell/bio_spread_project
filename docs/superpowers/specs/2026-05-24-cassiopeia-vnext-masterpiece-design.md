# Cassiopeia vNext Masterpiece Design

**Date:** 2026-05-24  
**Project:** Cassiopeia / dna-sentinel  
**Scope:** Three-model family: mini, medium, large  
**Status:** Design/specification; no implementation yet

---

## 1. Executive Thesis

Cassiopeia vNext will become a compact plasmid-risk model family built around one shared scientific idea:

> **Small models can compete with much larger DNA language models when the task is narrow, the inductive bias is biologically correct, and evaluation is leakage-resistant.**

The project will not try to imitate DNABERT as a general genome foundation model. DNABERT-2-class systems are roughly 117M parameters and trained on broad genomic corpora. Cassiopeia vNext will instead be a plasmid-specialist family that stays small while exploiting the geometry and biology of plasmids:

- plasmids are circular, not naturally linear;
- plasmid risk is multi-task: mobility, AMR cargo, and spread/expansion are coupled but not identical;
- k-mer histograms are fast but need spatial structure restored at the window level;
- inference models must be small, but training can use stronger teacher signals;
- every claim must survive held-out, reverse-complement, circular-shift, calibration, latency, and non-plasmid stress tests.

The resulting product identity:

> **Cassiopeia vNext: tiny models, sharp plasmid intelligence.**

---

## 2. Non-Negotiable Design Principles

1. **One family, three scales.**  
   Mini, medium, and large must share the same code path, model class, data pipeline, CLI, API, metrics, and report format. Only config values and optional training strategies differ.

2. **Small by construction.**  
   Model capacity must be budgeted, tested, and reported. A model that becomes powerful by quietly inflating size is considered a design failure.

3. **Biological bias over brute force.**  
   Improvements should encode plasmid-specific structure: circularity, multi-scale windows, motif locality, task coupling, and evidence localization.

4. **Inference stays simple.**  
   Teacher models, ensembles, PCGrad, and distillation can be training-time tools. Production inference must remain a single compact model.

5. **Evidence is a first-class output.**  
   The model must report task-specific evidence windows, not only scalar probabilities.

6. **No single-split hero numbers.**  
   Each model must be evaluated on validation, test, held-out test, non-plasmid controls, reverse-complement stress, circular-shift stress, calibration, latency, and parameter count.

7. **Scientific honesty.**  
   The project may claim domain-specialist competitiveness against large DNA models only on the plasmid-risk task, under the same data/evaluation protocol.

---

## 3. Shared Cassiopeia vNext Architecture

The three model variants use the same conceptual pipeline:

```text
Raw FASTA DNA
  -> canonical multi-scale k-mer features
  -> structural/window features
  -> deterministic/factorized projection
  -> circular plasmid positional encoding
  -> window-level multi-scale motif mixer
  -> compact sequence encoder
  -> task-conditioned adapters
  -> task-specific evidence pooling
  -> mobility / AMR / expansion heads
  -> calibrated probabilities + risk score + evidence
```

### 3.1 Input Features

The current feature extractor remains the base because it is fast and already validated:

- window sizes: 512, 2048, 8192 bp;
- default windows: 16 + 8 + 4 = 28;
- canonical reverse-complement-collapsed k-mers;
- current default k range: 4..6, 2728 dimensions;
- structural features: GC, GC skew, AT skew, dinucleotide composition.

vNext adds optional scale-specific feature profiles:

| Profile | k-mer range | Intended use |
|---|---:|---|
| compact | 4..6 | mini default; speed and compatibility |
| enriched | 3..7 | medium/large ablation; stronger motif spectrum |
| compact+shape | 4..6 + extra structural signals | production-safe default if enriched profile is too expensive |

The enriched profile must be gated by config and benchmarked. It is not automatically adopted unless it improves equal-weight champion score without unacceptable latency/size cost.

### 3.2 Circular Plasmid Positional Encoding (CPPE)

Current models treat window position mostly as sequence index plus scale embedding. vNext makes circularity explicit.

For each valid window, compute:

```text
phase = 2*pi*relative_position_within_scale
coord = [sin(phase), cos(phase), log(window_size), scale_id]
```

This coordinate is projected into hidden dimension and added to the projected window embedding.

Purpose:

- reduce FASTA cut-point bias;
- make circular shifts less disruptive;
- distinguish local windows from equivalent windows at different scales;
- improve biological plausibility without expensive base-level modeling.

### 3.3 Window-Level Multi-Scale Motif Mixer

k-mer histograms are efficient but lose order inside windows. Instead of expensive character-level CNNs, vNext restores local order at the **window embedding** level.

Add a lightweight module:

```text
DepthwiseSeparableConv1D(k=3)
DepthwiseSeparableConv1D(k=5)
DepthwiseSeparableConv1D(k=7)
Gated fusion
Residual LayerNorm
```

Purpose:

- capture neighboring-window motif transitions;
- approximate operon/backbone segment continuity;
- improve mobility signals such as conjugation/mobilization regions;
- add very few parameters relative to transformer-like alternatives.

### 3.4 Compact Sequence Encoder

The encoder remains small and CPU-friendly. Candidate blocks:

- GLUMixer with stronger masking semantics;
- optional low-rank attention only in medium/large;
- gated residual path with stochastic depth;
- factorized feed-forward expansion to avoid parameter blow-up.

A key rule: mini, medium, and large use the same block API so experiments are comparable.

### 3.5 Task-Conditioned Adapters

Each task receives its own adapter:

```text
shared window representation
  -> mobility adapter
  -> AMR adapter
  -> expansion adapter
```

Adapters remain bottlenecked. Capacity scales by model profile:

- mini: adapter rank 8;
- medium: adapter rank 12..16;
- large: adapter rank 24..32.

Purpose:

- prevent AMR/expansion from dominating mobility;
- allow task-specific evidence without duplicating the whole backbone;
- keep inference single-pass.

### 3.6 Task-Specific Evidence Pooling

Current output exposes mobility evidence only. vNext exposes all three:

```json
{
  "mobility_evidence": [...],
  "amr_evidence": [...],
  "expansion_evidence": [...]
}
```

Each task has an evidence scorer with shared implementation but task-specific parameters. Evidence is attention-like and must respect window masks.

The API prediction object should include:

- `top_mobility_windows`;
- `top_amr_windows`;
- `top_expansion_windows`;
- compact backward-compatible `top_windows` alias for mobility or overall risk.

### 3.7 Risk Score

Current risk score combines mobile probability, AMR probability, and expansion probability. vNext keeps this simple but makes it documented and configurable:

```text
risk_score = w_mob * P(mobile) + w_amr * P(AMR) + w_exp * P(expansion)
```

Default remains equal or near-equal unless product requirements say otherwise. Champion selection, however, uses equal task metrics, not risk score.

---

## 4. The Three Model Plans

## 4.1 Cassiopeia Mini: The Production Blade

### Purpose

Mini is the default deployable model: fast, robust, explainable, and small enough to run anywhere. It should be the model used by CLI/API unless the user explicitly asks for a larger profile.

### Budget

- target parameters: <= 500K;
- target checkpoint: <= 6 MB;
- target CPU inference: <= 5 ms/sequence on local Apple Silicon-class CPU for cached feature input, with end-to-end FASTA latency separately reported;
- feature profile: compact k=4..6 by default.

### Architecture

Mini should be a highly polished version of the current Cassiopeia:

```text
FRP/F-LoRA projection
  -> structural fusion
  -> CPPE
  -> one compact multi-scale window CNN
  -> 2 GLUMixer blocks
  -> rank-8 task adapters
  -> single-head task-specific evidence pooling
  -> task heads
```

### Training Strategy

Mini should not try to win by size. It wins by discipline:

1. leakage-resistant split only;
2. balanced task batches or class-balanced loss;
3. uncertainty-weighted multi-task objective;
4. moderate manifold mixup;
5. window dropout;
6. calibration on validation set;
7. optional distillation from medium/large once those exist.

### Scientific Angle

Mini demonstrates that a biologically biased model below 500K parameters can solve a plasmid-risk problem usually approached with much larger sequence models. Its strength is **efficiency with evidence**.

### Success Criteria

Mini is successful if it:

- beats the current checkpoint on equal-weight test or held-out score;
- stays below 500K trainable parameters;
- improves circular-shift stability;
- exposes task-specific evidence;
- keeps calibration acceptable, especially AMR/expansion ECE;
- remains the simplest production choice.

---

## 4.2 Cassiopeia Medium: The Main Champion

### Purpose

Medium is the primary research and publication candidate. It is the expected best trade-off between size, strength, latency, and maintainability.

### Budget

- target parameters: <= 1M;
- target checkpoint: <= 12 MB;
- target CPU inference: <= 10 ms/sequence for cached feature input;
- feature profile: compact+shape default; enriched k=3..7 as an ablation.

### Architecture

Medium scales the same core, not a different model:

```text
stronger F-LoRA / factorized projection
  -> richer structural fusion
  -> CPPE
  -> two multi-scale window CNN stages
  -> 3 compact sequence blocks
  -> rank-16 task adapters
  -> task-specific evidence pooling, 2 heads if ablation proves useful
  -> calibrated task heads
```

Medium may introduce a **micro-attention bridge** only if it remains cheap:

```text
low-rank token attention over 28 windows
```

This is not a full transformer. It exists to capture long-range plasmid backbone dependencies that convolution/mixers may miss.

### Training Strategy

Medium gets the full serious training protocol:

1. balanced sampler;
2. uncertainty-weighted task loss;
3. PCGrad or GradNorm behind config flag;
4. manifold mixup;
5. reverse-complement and circular-shift consistency regularization;
6. calibration;
7. distillation from an ensemble or large profile if available.

Consistency regularization is especially important:

```text
prediction(seq) ~= prediction(reverse_complement(seq))
prediction(seq) ~= prediction(circular_shift(seq))
```

These regularizers encode biological invariance directly and should impress reviewers more than raw parameter scaling.

### Scientific Angle

Medium is the flagship argument:

> A sub-1M plasmid-specialist model can approach or exceed much larger DNA language models on plasmid mobility/AMR/spread triage because it encodes plasmid geometry, multi-task causality, and evidence localization.

### Success Criteria

Medium is successful if it:

- becomes the best equal-weight score among mini/medium/large after size penalty;
- improves mobility without sacrificing AMR/expansion;
- gives stable RC/circular-shift outputs;
- stays below 1M parameters;
- remains simple enough to explain in a TÜBİTAK-style review panel.

---

## 4.3 Cassiopeia Large: The Compact Foundation Challenger

### Purpose

Large is the strongest domain-specialist model in the family. It is not the default production model. It is the benchmark, teacher, and publication-grade “how far can compact specialization go?” model.

### Budget

- target parameters: 3M to 5M maximum;
- target checkpoint: <= 60 MB;
- target CPU inference: reported, not hard-blocking, but should remain practical;
- feature profile: enriched k=3..7 and/or compact+shape selected by benchmark.

### Architecture

Large uses the full vNext architecture:

```text
factorized enriched projection
  -> structural and coverage fusion
  -> CPPE
  -> stacked multi-scale window CNN
  -> hybrid GLUMixer + low-rank attention encoder
  -> deeper task adapters
  -> multi-head task-specific evidence pooling
  -> calibrated task heads
```

Large may include a lightweight **cross-task reasoning layer** after task contexts are pooled:

```text
[mobility_context, amr_context, expansion_context]
  -> tiny gated task-graph module
  -> refined task logits
```

This must be small and interpretable. It should model that AMR, mobility, and expansion are biologically related while avoiding gradient shortcuts. Expansion may condition on detached mobility/AMR logits, as current Cassiopeia already does.

### Training Strategy

Large is trained as both model and teacher:

1. strong augmentation;
2. PCGrad enabled;
3. class-balanced objectives;
4. RC/circular-shift consistency;
5. stochastic depth;
6. early stopping by equal-weight score;
7. calibration;
8. teacher-logit export for mini/medium distillation.

Large can also support **model soup** experiments across seeds, but the shipped large profile should still be a single checkpoint unless an ensemble is explicitly documented as a teacher-only asset.

### Scientific Angle

Large is the panel-facing demonstration that the architecture scales:

> Even at 3–5M parameters, the model remains tens of times smaller than DNABERT-2-117M while incorporating plasmid-specific inductive bias, calibrated multi-task outputs, and window-level evidence.

### Success Criteria

Large is successful if it:

- obtains the highest raw equal-weight score or acts as the best teacher;
- stays below 5M parameters;
- gives strong held-out performance;
- improves mini/medium through distillation;
- remains explainable and auditable.

---

## 5. Equal-Weight Champion Protocol

The user explicitly selected equal task weighting. Champion score must therefore not overweight mobility.

Primary score:

```text
raw_task_score = mean(
  mobility_balanced_accuracy,
  AMR_AUROC,
  expansion_AUROC
)
```

Deployment-aware score:

```text
champion_score = raw_task_score - size_penalty - latency_penalty - instability_penalty
```

Where:

- `size_penalty` applies only when a model exceeds its declared profile budget;
- `latency_penalty` applies when profile latency targets are exceeded;
- `instability_penalty` applies for high reverse-complement or circular-shift prediction drift.

A model cannot be declared champion unless all required reports exist.

Required reports per profile:

```text
artifacts/cassiopeia_mini/report.json
artifacts/cassiopeia_medium/report.json
artifacts/cassiopeia_large/report.json
```

Each report must include:

- parameter count;
- checkpoint size;
- config hash;
- train/val/test/heldout sizes;
- val metrics;
- test metrics;
- heldout_test metrics;
- nonplasmid_control metrics;
- calibration metrics;
- reverse-complement stress;
- circular-shift stress;
- latency;
- final champion score.

---

## 6. Evaluation and Stress Tests

### 6.1 Standard Splits

Evaluate on:

- validation;
- test;
- heldout_test;
- nonplasmid_control.

### 6.2 Reverse-Complement Stress

For a sample of sequences:

```text
risk(seq) vs risk(reverse_complement(seq))
```

Report:

- max absolute risk difference;
- mean absolute risk difference;
- per-task probability drift.

### 6.3 Circular-Shift Stress

For plasmids, generate deterministic circular shifts:

```text
shift by 25%, 50%, 75% of sequence length
```

Report:

- mean absolute risk drift;
- max task probability drift;
- evidence-window stability after coordinate normalization.

### 6.4 Calibration

Report:

- Brier score;
- ECE;
- reliability by task.

Calibration is not optional. A model with strong AUROC but poor calibration must not be presented as deployment-ready.

### 6.5 Non-Plasmid Control

The non-plasmid control set should not be treated as a normal AUROC split if labels are single-class. It should instead report:

- false mobile rate;
- false AMR rate at configured threshold;
- false expansion rate at configured threshold;
- risk distribution quantiles.

---

## 7. Training System Plan

The current training code is compact but too monolithic for the next phase. Split only where it improves scientific control.

Target modules:

```text
src/dna_sentinel/training/losses.py
src/dna_sentinel/training/optim.py
src/dna_sentinel/training/calibration.py
src/dna_sentinel/training/evaluation.py
src/dna_sentinel/training/stress.py
src/dna_sentinel/training/distillation.py
```

Responsibilities:

- `losses.py`: focal BCE, CE with label smoothing, uncertainty weighting, distillation losses, consistency losses;
- `optim.py`: optimizer groups, scheduler, PCGrad/GradNorm utilities;
- `calibration.py`: temperature scaling and Platt scaling;
- `evaluation.py`: split metrics and champion scoring;
- `stress.py`: reverse-complement and circular-shift tests;
- `distillation.py`: teacher-logit export and student loss helpers.

The CLI should remain simple:

```bash
dna-sentinel train --config config/cassiopeia_mini.yaml
dna-sentinel train --config config/cassiopeia_medium.yaml
dna-sentinel train --config config/cassiopeia_large.yaml
dna-sentinel evaluate --checkpoint ... --data-dir data/dna_sentinel
dna-sentinel benchmark-family --configs config/cassiopeia_*.yaml
```

`benchmark-family` is optional in the first implementation pass but should be part of the final system.

---

## 8. Distillation Strategy

Distillation is the key to making small models unusually strong.

### 8.1 Teacher Sources

Allowed teachers:

1. Cassiopeia-large single checkpoint;
2. Cassiopeia-large seed soup or ensemble, teacher-only;
3. Cassiopeia-medium ensemble, teacher-only;
4. optional external DNABERT-2 embeddings/logits if a controlled offline experiment is added later.

External teacher integration must never become a hard runtime dependency.

### 8.2 Student Loss

Student training combines:

```text
supervised_task_loss
+ alpha * teacher_logit_KL
+ beta * evidence_alignment_loss
+ gamma * consistency_loss
```

Distillation should be profile-specific:

- mini: stronger distillation, because capacity is limited;
- medium: moderate distillation;
- large: no student distillation by default; it produces teacher signals.

### 8.3 Evidence Distillation

If teacher evidence is available, align student evidence distributions per task. This is scientifically attractive because the student learns not only output probabilities but also where to look.

---

## 9. Documentation and Product Polish

The final project should read like a serious research artifact and a usable engineering product.

Required documentation updates:

```text
README.md
MODEL_CARD.md
AI_AGENT_CONTEXT.md
docs/model_family.md
docs/evaluation_protocol.md
docs/api_contract.md
```

Documentation must clarify:

- product name: Cassiopeia;
- package/CLI name: dna-sentinel;
- mini/medium/large intended use;
- feature extraction method;
- model architecture;
- limitations;
- proper benchmark interpretation;
- why the model is not a general genome foundation model;
- how it can be compared to DNABERT-like systems on the plasmid-risk task.

---

## 10. Technical Debt to Fix During vNext

The following issues should be fixed as part of the implementation plan because they directly affect reproducibility and credibility:

1. `pyproject.toml` lacks complete dependencies, optional extras, and console script metadata even though installed egg-info has them.
2. Installed package metadata is stale (`0.1.0`) while `pyproject.toml` says `0.2.0`.
3. Dockerfile does not copy artifacts but default command expects a checkpoint path under `artifacts/`.
4. README says FRP has zero storage, while current checkpoint includes the FRP buffer.
5. AI agent context says `scale_ids` are unused, while current code uses `scale_embed` when `scale_ids` are passed.
6. API docs should consistently use `/predict-batch`, matching code.
7. Artifact directory contains many legacy experiments; final reports need a clean current-family namespace.

---

## 11. Phased Delivery Plan

### Phase 0: Measurement Foundation

Goal: make the current system measurable and reproducible before changing architecture.

Deliverables:

- baseline report for current checkpoint;
- parameter count utility;
- checkpoint size utility;
- latency benchmark;
- RC stress test;
- circular-shift stress test;
- non-plasmid false-positive report.

### Phase 1: Config Family

Goal: introduce mini/medium/large as first-class profiles.

Deliverables:

- `config/cassiopeia_mini.yaml`;
- `config/cassiopeia_medium.yaml`;
- `config/cassiopeia_large.yaml`;
- model factory support;
- tests enforcing parameter budgets.

### Phase 2: vNext Model Core

Goal: implement architecture upgrades without breaking CLI/API.

Deliverables:

- CPPE module;
- window-level multi-scale CNN module;
- improved encoder config;
- task-specific evidence pooling;
- backward-compatible prediction object.

### Phase 3: Training Hardening

Goal: make optimization worthy of the architecture.

Deliverables:

- balanced sampling/loss;
- optional PCGrad;
- consistency losses;
- calibration module;
- structured history/report output.

### Phase 4: Distillation

Goal: use large/ensemble knowledge to strengthen small models.

Deliverables:

- teacher-logit export;
- student training support;
- optional evidence distillation;
- mini/medium distilled runs.

### Phase 5: Family Benchmark

Goal: select the real champion honestly.

Deliverables:

- reports for mini, medium, large;
- equal-weight ranking;
- latency/size/stability penalties;
- final model card update.

### Phase 6: Product Polish

Goal: make the repo presentation match the science.

Deliverables:

- cleaned README;
- updated MODEL_CARD;
- Docker fix;
- package metadata fix;
- API contract docs;
- legacy artifact notes.

---

## 12. Review-Panel Framing

For a TÜBİTAK-style jury, the strongest framing is:

1. **Problem importance:** plasmids drive antimicrobial resistance spread; fast risk triage matters.
2. **Scientific gap:** large genome language models are powerful but heavy and generic; plasmid risk has domain-specific structure.
3. **Technical idea:** encode plasmid circularity, multi-scale sequence windows, task-coupled risk, and evidence localization into compact neural models.
4. **Innovation:** sub-5M specialist family with task-specific evidence and biological invariance tests.
5. **Rigor:** leakage-aware splits, held-out data, non-plasmid controls, calibration, RC/circular-shift stress, latency, size reporting.
6. **Impact:** deployable model family from edge-speed mini to research-grade large, with the ability to distill stronger models into production models.

The jury-facing one-liner:

> **Cassiopeia vNext is a plasmid-specialized DNA intelligence family that uses biological structure, not brute-force scale, to approach foundation-model performance in a deployable footprint.**

---

## 13. Out of Scope for First Implementation

The following are explicitly out of scope unless later approved:

- online dependency on DNABERT or Hugging Face models during inference;
- requiring BLAST, annotation, or metadata for prediction;
- replacing the entire feature pipeline with base-level transformers;
- shipping an ensemble as the default production model;
- claiming general genome foundation-model superiority.

---

## 14. Final Recommendation

Build all three profiles, but treat them as a coordinated system:

- **mini** is the production blade;
- **medium** is the main champion;
- **large** is the compact foundation challenger and teacher.

The most elegant path is not to make one giant model. It is to make a family where large teaches, medium competes, and mini deploys.
