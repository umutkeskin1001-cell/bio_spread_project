# BioSpread: Plazmid Coğrafi Yayılım Erken Uyarısı

## Problem
Antimikrobiyal direnç taşıyan plazmid omurgaları farklı ülke ve konaklara yayıldığında halk sağlığı açısından takip edilmesi zor bir risk oluşur. Bu proje, geçmiş gözlemlerden yakın vadeli coğrafi yayılım riskini tahmin eder.

## Kurulum
Seçilen birincil model: `geobio_reliability_ensemble`.
Validation / doğrulama modu: `spatial_group_cv_stacked`.
Eğitim gözlemleri: `2020` ve öncesi.
Değerlendirme ufku: sonraki `3` yıl.

## Validation And Reliability
- ROC AUC: `0.916`
- Average precision: `0.887`
- Pozitif prevalans: `0.366`
- Top-k precision: `1.000`
- Abstain/review oranı: `0.078`
- Kalibrasyon hatası: `0.058`
- Brier score: `0.113`
- Group OOF ROC AUC: `0.916`
- Temporal holdout ROC AUC: `0.885`
- Bootstrap ROC AUC CI: `[0.895, 0.933]`
- Max single-feature AUC: `0.914` (suspicious: `0`)
- Leakage guard: `not_checked`
- Kalite kapıları: `review`
- Katsayı özeti: `surv_intensity=0.021; host_sampling_shannon=0.013; saturation_deficit=0.011; reach_potential=0.011; gnn_embed_6=0.009`

## Calibration
| Bin | Mean prediction | Observed rate | Count |
| --- | ---: | ---: | ---: |
| 0.000-0.200 | 0.096 | 0.068 | 444 |
| 0.200-0.400 | 0.292 | 0.190 | 126 |
| 0.400-0.600 | 0.492 | 0.416 | 77 |
| 0.600-0.800 | 0.709 | 0.552 | 87 |
| 0.800-1.000 | 0.944 | 0.894 | 255 |

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
| 2 | AA304 | 1.000 | high | 14 | prob=1.00; knownness=0.76 |
| 3 | AA038 | 1.000 | high | 35 | prob=1.00; knownness=0.77 |
| 4 | AA372 | 1.000 | high | 39 | prob=1.00; knownness=0.75 |
| 5 | AA345 | 1.000 | high | 12 | prob=1.00; knownness=0.76 |
| 6 | AA336 | 1.000 | high | 25 | prob=1.00; knownness=0.77 |
| 7 | AA378 | 1.000 | high | 35 | prob=1.00; knownness=0.49 |
| 8 | AB685 | 1.000 | high | 28 | prob=1.00; knownness=0.65 |
| 9 | AA919 | 1.000 | high | 10 | prob=1.00; knownness=0.68 |
| 10 | AA282 | 1.000 | high | 23 | prob=1.00; knownness=0.75 |

## Ana Projeden Farkı
Bu bağımsız proje genel plazmid önceliklendirme platformunu değil, tek bir biyolojik soruyu hedefler: plazmid omurgalarının coğrafi yayılım riskini erken saptamak.
