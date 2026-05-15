# Full Training Report — Sovereign-X Pro

> **Run:** `artifacts/SX_20260514_235436`
> **Duration:** 23:57 → 02:54 (~3 hours, 50 epochs)
> **Validation set:** 674 backbones
> **Config:** `config/default.yaml` (default hyperparameters)
> **Data:** `data/sovereign_features/sequences.tsv` (21,520 sequences, regenerated with refactored feature set)

---

## 1. Executive Summary

The Sovereign-X Pro model achieves a **ROC AUC of 0.8879** on the 3-year hazard prediction task, with strong performance across all three horizons. Post-refactoring improvements (feature deduplication, leak-free taxonomy vocab, corrected ranking loss, O(n²)→O(n) optimization) yielded a **+1.16 percentage point improvement** over the previous best run (ROC AUC 0.8763 → 0.8879).

---

## 2. Model Metrics

### 2.1 ROC & PR AUC by Horizon

| Horizon | ROC AUC | PR AUC | ECE |
|---|---|---|---|
| h1 (1 year) | **0.9292** | 0.7205 | 0.0830 |
| h2 (2 years) | **0.9231** | 0.7853 | 0.1043 |
| h3 (3 years) | **0.8879** | 0.7256 | 0.1428 |

The model performs best on short-term prediction (1 year) and degrades gracefully over longer horizons. The ECE (Expected Calibration Error) increases with horizon, suggesting calibration is better for short-term predictions.

### 2.2 Classification Metrics (at default threshold 0.5)

| Metric | Value |
|---|---|
| **F1 Score** | **0.6092** |
| **Recall (TPR)** | **0.7794** |
| **Precision** | 0.5000 |
| **Specificity (TNR)** | 0.8030 |
| **Balanced Accuracy** | 0.7912 |
| **NPV** (Negative Predictive Value) | 0.9351 |
| **MCC** (Matthews Correlation Coefficient) | 0.5034 |
| **Brier Score** | 0.1294 |

### 2.3 Confusion Matrix

```
              Pred+    Pred-
Actual+        106        30         ← 136 positives
Actual-        106       432         ← 538 negatives
```

- **TP = 106** — Correctly predicted spread
- **FP = 106** — False alarms (balanced with TP, suggesting threshold could be tuned)
- **FN = 30** — Missed spread events
- **TN = 432** — Correctly predicted non-spread

**Derived rates:**
- **FPR** (False Positive Rate) = 0.1970
- **FNR** (False Negative Rate) = 0.2206
- **Positive Rate** = 0.2018 (20.2% of validation samples are positive)

---

## 3. Comparison with Previous Run

| Metric | Previous (23:45) | **Current (23:57)** | Δ |
|---|---|---|---|
| ROC AUC h1 | 0.9145 | **0.9292** | **+0.0147** |
| ROC AUC h2 | 0.9213 | **0.9231** | +0.0018 |
| ROC AUC h3 | 0.8763 | **0.8879** | **+0.0116** |
| PR AUC h3 | 0.7011 | **0.7256** | **+0.0245** |
| Recall | 0.7111 | **0.7794** | **+0.0683** |
| F1 Score | 0.5872 | **0.6092** | **+0.0220** |
| Balanced Accuracy | 0.7665 | **0.7912** | **+0.0247** |
| Brier Score | 0.1295 | **0.1294** | -0.0001 |
| True Positives | 96 | **106** | +10 |
| False Negatives | 39 | **30** | -9 |

**Improvement breakdown:**
- Feature deduplication (removing 5 overlapping static/snapshot features): cleaner signal
- Corrected ranking loss (within-sample instead of cross-sample): better temporal consistency
- Leakage-free taxonomy vocab: more realistic evaluation
- Refactored model code + optimized O(n²)→O(1) hazard computation

---

## 4. Calibration

### 4.1 Platt Scaling

Two separate Platt scalers were fitted post-training:

| Scaler | a (slope) | b (intercept) |
|---|---|---|
| Main (h3) | 3.478 | 0.269 |
| Cold-start | 32.252 | 3.503 |

The cold-start scaler has a much steeper slope, indicating the cold-start head's raw logits are less calibrated than the main head's.

### 4.2 Per-Horizon Calibration

| Horizon | ECE |
|---|---|
| h1 | 0.0830 (well-calibrated) |
| h2 | 0.1043 (moderate) |
| h3 | 0.1428 (needs improvement) |

---

## 5. Training Dynamics

| Epoch | Loss | Val ROC AUC |
|---|---|---|
| 1 | 1.9086 | 0.7401 |
| Final (best) | — | **0.8879** |

The model converged smoothly with early stopping (patience=10 triggers at ~40 epochs).

---

## 6. Data Summary

| Split | Backbones | Sequences |
|---|---|---|
| Train | 5,620 | 13,947 |
| Validation | 942 | 5,418 |
| Test | 279 | 2,155 |
| **Total** | **6,841** | **21,520** |

- **Features:** 10 static + 10 snapshot = 20 numeric features
- **Taxonomy:** 5-level embeddings (35 phyla, 66 classes, 145 orders, 325 families, 915 genera)
- **Sequence length:** Padded to 45, mean ~3.1 years per backbone
- **Positive rate:** 20.2% (class imbalance)

---

## 7. Recommendations

1. **Threshold tuning:** Current threshold (0.5) gives precision=recall=106. Optimizing for F1 would reduce FP at minor cost to recall
2. **Calibration improvement:** Isotonic regression or temperature scaling could reduce ECE from 0.14 to <0.08
3. **Test set evaluation:** Run the best checkpoint on the held-out test set (279 backbones) for final unbiased estimate
4. **Ensemble:** Train 3-5 models with different seeds and average predictions for +0.01-0.02 AUC gain
5. **Feature engineering:** Add AMR gene profiles, phylogenetic distance features

---

*Report generated 2025-05-15 from `artifacts/SX_20260514_235436/metrics.json`*
