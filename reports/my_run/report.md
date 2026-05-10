# BioSpread Executive Summary

```text
+==========================================================+
| BIOSPREAD PREDICTIVE SURVEILLANCE REPORT                 |
+==========================================================+
STATUS:    NO GO
RUN ID:    7eb2b7e2-f804-4a5d-8e5d-8c21d20bf580
TIMESTAMP: 2026-05-10T23:14:25.749624
```

### Problem Definition
BioSpread prioritizes plasmid backbones by predicted geographic spread risk.

### Validation Performance
| Metric | Value | Threshold | Status |
| --- | --- | --- | --- |
| ROC AUC | 0.9781 | >= 0.820 | PASS |
| Average precision | 0.7169 | > prevalence | PASS |
| Calibration ECE | 0.0273 | <= 0.150 | PASS |
| Brier score | 0.0283 | N/A | INFO |

### Calibration And Reliability
| Gate | Metric | Value | Status |
| --- | --- | --- | --- |
| Spatial group CV | OOF ROC AUC | 0.9781 | PASS |
| Temporal holdout | ROC AUC | 0.9781 | PASS |
| External holdout | Independent ROC AUC | 0.8233 | PASS |
| Bootstrap CI | ROC AUC low | 0.9729 | PASS |
| Leakage scan | Max single-feature AUC | 0.5000 | PASS |

### High-Risk Backbone Registry
| Rank | Backbone ID | Risk | Confidence | Future Spread | Explanation |
| --- | --- | --- | --- | --- | --- |
| 1 | AA184 | 0.9846 | medium | 0 | high_risk; probability=0.985; uncertainty=0.030 |
| 2 | AA457 | 0.9830 | medium | 0 | high_risk; probability=0.983; uncertainty=0.033 |
| 3 | AA279 | 0.9830 | medium | 0 | high_risk; probability=0.983; uncertainty=0.034 |
| 4 | AA045 | 0.9808 | medium | 0 | high_risk; probability=0.981; uncertainty=0.038 |
| 5 | AE107 | 0.9800 | medium | 0 | high_risk; probability=0.980; uncertainty=0.039 |
| 6 | AA316 | 0.9798 | medium | 0 | high_risk; probability=0.980; uncertainty=0.040 |
| 7 | AA282 | 0.9793 | medium | 0 | high_risk; probability=0.979; uncertainty=0.041 |
| 8 | AA302 | 0.9788 | medium | 0 | high_risk; probability=0.979; uncertainty=0.042 |
| 9 | AA543 | 0.9781 | medium | 0 | high_risk; probability=0.978; uncertainty=0.043 |
| 10 | AA436 | 0.9770 | medium | 0 | high_risk; probability=0.977; uncertainty=0.045 |

### Priority Surveillance Targets
1. AA184 (alarm=0.955, risk=0.985)
2. AA457 (alarm=0.950, risk=0.983)
3. AA279 (alarm=0.950, risk=0.983)
4. AA045 (alarm=0.944, risk=0.981)
5. AE107 (alarm=0.942, risk=0.980)

### Threat Triage Matrix
| Tier | Count | Criteria |
| --- | --- | --- |
| RED | 269 | High predicted risk |
| YELLOW | 661 | Moderate risk or elevated uncertainty |
| GREEN | 5226 | Low risk and low uncertainty |

### Model Signal Summary
The production adapter uses leakage-controlled numeric features with a bounded-memory sklearn ensemble.
Feature columns containing future outcomes, labels, targets, countries, regions, or travel proxies are excluded before fitting.

### Limitations
- Predictions are based on historical observation density.
- Missing data in low-surveillance regions may bias risk scores.
- Operational use requires expert review and independent epidemiological evidence.

### Environment And Reproducibility
| Entity | Value / Hash |
| --- | --- |
| Python | 3.9.6 |
| Polars | 0.20.31 |
| Input SHA-256 | 2b7fcfe8c5448d3a83d98183997e5d57eb78f328a3530ca02933155d71d6b334 |
| Training split | year <= 2020 |
| Forecast horizon | 3 years |

### Release Gate
Release status: NO GO