# BioSpread: Plazmid Coğrafi Yayılım Erken Uyarısı

## Problem
Antimikrobiyal direnç taşıyan plazmid omurgaları farklı ülke ve konaklara yayıldığında halk sağlığı açısından takip edilmesi zor bir risk oluşur. Bu proje, geçmiş gözlemlerden yakın vadeli coğrafi yayılım riskini tahmin eder.

## Kurulum
Seçilen birincil model: `geobio_reliability_ensemble`.
Validation / doğrulama modu: `spatial_group_cv_stacked`.
Eğitim gözlemleri: `2020` ve öncesi.
Değerlendirme ufku: sonraki `3` yıl.

## Validation And Reliability
- ROC AUC: `0.883`
- Average precision: `0.838`
- Pozitif prevalans: `0.366`
- Top-k precision: `1.000`
- Abstain/review oranı: `0.091`
- Kalibrasyon hatası: `0.046`
- Brier score: `0.128`
- Group OOF ROC AUC: `0.883`
- Temporal holdout ROC AUC: `0.880`
- Bootstrap ROC AUC CI: `[0.862, 0.905]`
- Max single-feature AUC: `0.919` (suspicious: `0`)
- Leakage guard: `not_checked`
- Kalite kapıları: `review`
- Katsayı özeti: `gnn_embed_2=0.021; surv_intensity=0.021; gnn_embed_5=0.015; host_sampling_shannon=0.010; reach_potential=0.008`

## Calibration
| Bin | Mean prediction | Observed rate | Count |
| --- | ---: | ---: | ---: |
| 0.000-0.200 | 0.075 | 0.099 | 454 |
| 0.200-0.400 | 0.289 | 0.246 | 142 |
| 0.400-0.600 | 0.488 | 0.378 | 90 |
| 0.600-0.800 | 0.708 | 0.646 | 82 |
| 0.800-1.000 | 0.943 | 0.882 | 221 |

## Leakage And Audit
Feature columns are checked against future/outcome naming patterns, and single-feature AUC is monitored to catch near-deterministic leakage.

## Release Gate
Release readiness is determined from quality gates, drift checks, and model-registry trend evidence. Fresh output directories usually start as conditional_go until enough registry history exists.

## Limitations
This is a retrospective early-warning benchmark over packaged data. It is not clinical diagnosis, a patient-level decision system, or proof of field deployment performance.

## Reproducibility
The run writes input hashes, selected input mode, threshold sources, environment versions, model registry entries, and release-gate artifacts.

## En Riskli Adaylar
| Sıra | Backbone | Risk | Güven | Yeni ülke | Açıklama |
| --- | --- | ---: | --- | ---: | --- |
| 1 | AA840 | 1.000 | high | 9 | prob=1.00; knownness=0.76 |
| 2 | AA919 | 1.000 | high | 10 | prob=1.00; knownness=0.68 |
| 3 | AA038 | 1.000 | high | 35 | prob=1.00; knownness=0.77 |
| 4 | AA336 | 1.000 | high | 25 | prob=1.00; knownness=0.77 |
| 5 | AA739 | 0.999 | high | 36 | prob=1.00; knownness=0.83 |
| 6 | AA372 | 0.999 | high | 39 | prob=1.00; knownness=0.75 |
| 7 | AB685 | 0.999 | high | 28 | prob=1.00; knownness=0.65 |
| 8 | AA304 | 0.999 | high | 14 | prob=1.00; knownness=0.76 |
| 9 | AB595 | 0.999 | high | 24 | prob=1.00; knownness=0.76 |
| 10 | AA551 | 0.999 | high | 17 | prob=1.00; knownness=0.85 |

## Ana Projeden Farkı
Bu bağımsız proje genel plazmid önceliklendirme platformunu değil, tek bir biyolojik soruyu hedefler: plazmid omurgalarının coğrafi yayılım riskini erken saptamak.
