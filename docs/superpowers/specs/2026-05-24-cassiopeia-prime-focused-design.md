# Cassiopeia Prime Focused Design

**Date:** 2026-05-24  
**Project:** Cassiopeia / dna-sentinel  
**Scope:** One compact champion model, not a three-model family  
**Target size:** ~1M parameters maximum  
**Status:** Revised design/specification; no implementation yet

---

## 1. Scope Correction

The previous three-profile plan was too broad for the current repository and team size. The project currently has roughly 1,100 lines of core source code and a compact, working training/evaluation stack. A six-phase family roadmap with mini/medium/large profiles, distillation, Docker/package polish, multiple stress suites, and six new training modules would create more process than progress.

Cassiopeia Prime narrows the goal:

> **Build one excellent compact plasmid-risk model around ~1M parameters, with the highest practical gain per line of code changed.**

The model should improve the existing Cassiopeia without turning the repository into a research platform. The winning path is not architectural maximalism. The winning path is a small number of biologically meaningful additions plus stronger training discipline.

---

## 2. Final Product Thesis

Cassiopeia Prime is a single plasmid-specialist model that predicts:

1. mobility class;
2. AMR cargo probability;
3. expansion/spread probability;
4. task-specific evidence windows.

It remains DNA-only and annotation-free:

```text
FASTA DNA -> k-mer/structural windows -> compact neural model -> calibrated risks + evidence
```

The project claim should be precise:

> **Cassiopeia Prime is not a general DNA foundation model. It is a compact, biologically biased plasmid-risk specialist designed to compete with much larger sequence models on this specific triage task.**

Motto:

> **One compact model. Maximum plasmid signal.**

---

## 3. What We Explicitly Drop

The revised plan removes the following from the first implementation:

1. **No mini/medium/large family.**  
   Build one model first. Add variants only after one champion is demonstrably maxed out.

2. **No distillation.**  
   A 4M teacher to 500K student gap is not large enough to justify early complexity. Distillation can return later if an actually stronger teacher exists.

3. **No evidence distillation.**  
   It is speculative at this stage and would be hard to validate cleanly.

4. **No early training-package refactor.**  
   `train.py` is about 290 lines. Splitting it into six modules now would increase coordination overhead. Refactor only after the new approach works.

5. **No cross-task reasoning layer in v1.**  
   Current expansion conditioning on detached mobility/AMR logits is enough for now.

6. **No broad product polish in the modeling pass.**  
   Docker/package metadata/docs can be fixed later unless they block experiments.

7. **No large architectural zoo.**  
   Avoid stacking CPPE, CNN, micro-attention, cross-task reasoning, distillation, and new features all at once. Each addition must have a plausible effect and a test.

---

## 4. Technical Diagnosis

Current Cassiopeia is already strong for its size:

- roughly 419K trainable parameters;
- held-out AMR AUROC above 0.93 in current local evaluation;
- held-out expansion AUROC above 0.82;
- mobility is useful but still the hardest and most fragile signal;
- core feature representation is multi-scale canonical k-mer windows.

The most likely performance ceiling is not insufficient layer count. It is the information bottleneck and invariance behavior of the k-mer feature regime:

- k-mer histograms are fast but lose local order;
- plasmids are circular but FASTA files are linearized;
- reverse-complement and circular-shift invariance must be enforced, not assumed;
- class/task imbalance can cause easy tasks to dominate training;
- evidence output is currently less complete than the multi-task objective.

Therefore, the revised plan prioritizes:

1. **training improvements first;**
2. **robustness regularization second;**
3. **small, targeted architecture additions third;**
4. **measurement before expansion.**

---

## 5. Cassiopeia Prime Architecture

Target: one model near but below **1M parameters**.

The architecture should remain close to the current codebase:

```text
k-mer features [B, 28, 2728]
  -> FRP + F-LoRA projection
  -> structural feature fusion
  -> circular plasmid positional encoding
  -> lightweight window motif convolution
  -> compact GLUMixer stack
  -> task-conditioned adapters
  -> task-specific evidence pooling
  -> task heads
```

### 5.1 Keep the Existing Backbone Philosophy

Preserve:

- canonical k-mer extractor;
- 28-window multi-scale layout;
- structural features;
- FRP/F-LoRA projection idea;
- GLUMixer-style compact encoder;
- detached mobility/AMR conditioning for expansion;
- calibration after training;
- CLI/API-compatible prediction path.

This minimizes implementation risk and keeps comparisons fair.

### 5.2 Add CPPE, But Keep It Tiny

Add circular plasmid positional encoding as a small module:

```text
coord = [sin(theta), cos(theta), scale_norm]
coord_embedding = Linear(3 -> hidden_dim)
x = x + coord_embedding
```

Where `theta` is computed per window within its scale group.

Purpose:

- reduce FASTA cut-point bias;
- help circular-shift consistency;
- encode plasmid biology with negligible parameter cost.

This is one of the few architecture changes worth making because it directly matches plasmid geometry.

### 5.3 Add One Window-Level Motif Convolution

Add a single lightweight depthwise separable convolution block over the 28 window embeddings:

```text
x_conv = depthwise_conv_1d(x, kernel=3 or 5)
x = LayerNorm(x + pointwise_gate(x_conv))
```

Do not add a large multi-kernel stack initially. Start with one simple block. If ablation supports it, extend to kernels 3/5.

Purpose:

- restore some neighboring-window order lost by histogram features;
- capture local genomic segment continuity;
- keep latency and parameter count low.

### 5.4 Do Not Add Micro-Attention Initially

Micro-attention may help, but it overlaps with GLUMixer and convolution. It should not be in the first Prime pass. Add only if CPPE + convolution + training improvements plateau.

### 5.5 Task-Specific Evidence Is Required

Current prediction exposes mobility evidence. Prime should expose evidence per task:

```text
mobility_evidence
amr_evidence
expansion_evidence
```

The pooling implementation should remain compact:

- one scorer per task;
- masked softmax over windows;
- weighted context per task;
- top-k windows in prediction output.

This is important because it improves scientific credibility and user value without requiring a larger model.

### 5.6 Parameter Budget

Initial target config:

```yaml
hidden_dim: 160 or 192
frp_out_dim: 320 or 384
n_layers: 3
lora_rank: 12 or 16
adapter_rank: 12 or 16
n_evidence_heads: 1
window_conv: true
cppe: true
```

The implementation must include a parameter-count test:

```text
Cassiopeia Prime trainable parameters <= 1,000,000
```

If 192 hidden exceeds budget after evidence changes, use 160. The target is not to hit 1M exactly. The target is to spend capacity where it matters.

---

## 6. Training Plan: Main Source of Gains

Training changes are expected to produce more value than architectural additions.

### 6.1 Balanced Sampling

Current training shuffles uniformly. Prime should add class-aware or task-aware sampling without overengineering.

Minimum viable approach:

- compute sample weights from mobility class, AMR label, and expansion label;
- use a `WeightedRandomSampler` or weighted index draw;
- keep implementation in `train.py` until it becomes unwieldy.

Purpose:

- prevent majority mobility classes from dominating;
- keep AMR/expansion positives visible;
- improve equal-weight task score without manually overweighting a task metric.

### 6.2 Consistency Regularization

Add training-time consistency on transformed sequences/features.

Two invariances matter:

1. reverse-complement invariance;
2. circular-shift invariance.

Because the training loop currently operates on precomputed features, the first practical version should be feature-level and/or dataset-preprocessing aware. Two options:

**Option A: feature-pair cache**  
Precompute RC and circular-shift features for training records and load them optionally.

**Option B: online sequence transform path**  
Use raw JSONL records during augmentation to regenerate transformed features for a fraction of batches.

Recommendation for first implementation: **Option A** if storage is acceptable. It is simpler, deterministic, and faster during training.

Consistency loss:

```text
KL(task_probs(original), task_probs(transformed))
```

Apply with small weight, e.g. `0.05..0.2`, and tune by validation score.

### 6.3 Keep WindowDropout and Manifold Mixup

Keep:

- WindowDropout;
- manifold mixup;
- stochastic depth;
- uncertainty-weighted task loss;
- focal BCE for binary tasks;
- label smoothing for mobility.

Do not add PCGrad in the first Prime implementation unless task gradients show actual conflict. PCGrad is a reasonable second-stage addition, not mandatory v1 complexity.

### 6.4 Calibration Remains Required

Keep Platt/temperature calibration. Do not make it a new subsystem yet.

Report:

- AUROC/AUPRC;
- balanced accuracy;
- Brier;
- ECE.

A stronger model with worse calibration is not automatically better.

---

## 7. Evaluation Plan

The evaluation suite should be focused but credible.

### 7.1 Required Metrics

Evaluate current baseline and Prime on:

- validation;
- test;
- heldout_test;
- nonplasmid_control.

Primary score uses equal task weighting:

```text
score = mean(
  mobility_balanced_accuracy,
  amr_auroc,
  expansion_auroc
)
```

No hidden task weighting.

### 7.2 Robustness Checks

Add two targeted stress checks:

1. reverse-complement drift;
2. circular-shift drift.

Report:

```text
mean_abs_probability_drift_by_task
max_abs_probability_drift_by_task
mean_abs_risk_drift
max_abs_risk_drift
```

These checks matter because they verify the biological invariances the model claims to encode.

### 7.3 Non-Plasmid Control Reporting

Do not report AUROC as meaningful when the set is single-class. Report:

- false mobile rate;
- false AMR rate;
- false expansion rate;
- risk score quantiles;
- mean predicted risk.

### 7.4 Benchmark Output

Add one compact report file for the experiment:

```text
artifacts/cassiopeia_prime/report.json
```

Required fields:

- parameter count;
- checkpoint size;
- config;
- split metrics;
- robustness metrics;
- non-plasmid false-positive metrics;
- latency if easy to measure.

Latency can be a simple local benchmark; it does not need a full benchmarking framework.

---

## 8. Implementation Phases

This is now a focused solo-dev plan, not a 6-month platform rewrite.

### Phase 1: Baseline and Measurement

Goal: freeze the current baseline.

Deliverables:

- current checkpoint report;
- parameter count helper;
- equal-weight score calculation;
- non-plasmid false-positive report;
- simple RC/circular-shift evaluation helper.

This phase should touch minimal code.

### Phase 2: Model Upgrade

Goal: implement one Prime model under 1M parameters.

Deliverables:

- config additions for CPPE/window conv/adapter rank;
- CPPE module;
- one lightweight window convolution block;
- task-specific evidence pooling;
- parameter-budget test;
- prediction output updated with task-specific evidence while preserving backward compatibility.

### Phase 3: Training Upgrade

Goal: extract most gains from better optimization, not bigger architecture.

Deliverables:

- balanced sampling;
- consistency regularization using precomputed transform features or a deterministic feature-pair cache;
- training config flags;
- validation score comparison against baseline.

### Phase 4: Final Benchmark and Documentation

Goal: prove whether Prime is better.

Deliverables:

- trained Prime checkpoint;
- `artifacts/cassiopeia_prime/report.json`;
- updated README/MODEL_CARD only for claims backed by metrics;
- decision note: keep Prime, tune further, or revisit larger variants.

---

## 9. Files Expected to Change

Keep changes concentrated.

Likely modified files:

```text
src/dna_sentinel/model.py
src/dna_sentinel/train.py
src/dna_sentinel/utils.py
src/dna_sentinel/cli.py
src/dna_sentinel/features.py       # only if transform-feature cache needs support
config/dna_sentinel.yaml           # or new config/cassiopeia_prime.yaml
tests/test_model.py
tests/test_train.py
tests/test_features.py             # only if feature cache support is added
tests/test_cli_predict.py          # only if prediction schema changes visibly
README.md                          # final phase only
MODEL_CARD.md                      # final phase only
```

Avoid new package/module trees unless a file becomes genuinely hard to maintain.

---

## 10. Success Criteria

Cassiopeia Prime is successful if all are true:

1. trainable parameters are <= 1,000,000;
2. all existing tests pass;
3. new parameter-budget and evidence-output tests pass;
4. equal-weight score improves over current checkpoint on at least heldout_test or test without severe degradation on the other;
5. reverse-complement and circular-shift drift improve or remain very low;
6. task-specific evidence is available in API/CLI prediction objects;
7. implementation remains compact enough that the repo still feels like Cassiopeia, not a framework.

---

## 11. Deferred Ideas

The following are valid but explicitly deferred:

- medium/large model family;
- distillation;
- evidence distillation;
- PCGrad by default;
- low-rank attention;
- cross-task reasoning layers;
- full training package refactor;
- Docker/package metadata polish unless blocking;
- external DNABERT teacher integration.

These ideas should return only if Prime plateaus after the focused training and robustness upgrades.

---

## 12. Final Recommendation

Build **one** model: Cassiopeia Prime.

The model should be roughly 1M parameters, biologically biased, robust to plasmid symmetries, and supported by clean evidence outputs. The implementation should be intentionally small:

```text
CPPE
+ one window convolution
+ task-specific evidence
+ balanced sampling
+ consistency regularization
+ honest benchmark report
```

This is the highest-leverage path from the current codebase to a stronger, more credible model.
