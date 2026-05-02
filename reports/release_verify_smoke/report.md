# BioSpread: Plazmid Coğrafi Yayılım Erken Uyarısı

## Problem
Antimikrobiyal direnç taşıyan plazmid omurgaları farklı ülke ve konaklara yayıldığında halk sağlığı açısından takip edilmesi zor bir risk oluşur. Bu proje, geçmiş gözlemlerden yakın vadeli coğrafi yayılım riskini tahmin eder.

## Kurulum
Seçilen birincil model: `mobility`.
Validation / doğrulama modu: `cross_validated`.
Eğitim gözlemleri: `2020` ve öncesi.
Değerlendirme ufku: sonraki `3` yıl.

## Validation And Reliability
- ROC AUC: `0.500`
- Average precision: `0.500`
- Pozitif prevalans: `0.400`
- Top-k precision: `0.400`
- Abstain/review oranı: `0.000`
- Kalibrasyon hatası: `0.437`
- Brier score: `0.290`
- Group OOF ROC AUC: `not_evaluated`
- Temporal holdout ROC AUC: `not_evaluated`
- Bootstrap ROC AUC CI: `[0.500, 0.500]`
- Max single-feature AUC: `0.000` (suspicious: `0`)
- Leakage guard: `not_checked`
- Kalite kapıları: `review`
- Katsayı özeti: `clinical=0.356; country=0.329; mobility=0.307; amr=0.221; low_knownness=0.086; host=0.000`

## Calibration
| Bin | Mean prediction | Observed rate | Count |
| --- | ---: | ---: | ---: |
| 0.000-0.200 | 0.141 | 0.500 | 2 |
| 0.200-0.400 | 0.367 | 0.000 | 1 |
| 0.400-0.600 | 0.585 | 1.000 | 1 |
| 0.600-0.800 | 0.684 | 0.000 | 1 |
| 0.800-1.000 | not_evaluated | not_evaluated | 0 |

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
| 1 | bb_alpha | 0.789 | medium | 3 | prob=0.79; knownness=0.33 |
| 2 | bb_gamma | 0.575 | review | 1 | prob=0.57; knownness=0.23 |
| 3 | bb_delta | 0.549 | review | 3 | prob=0.55; knownness=0.16 |
| 4 | bb_epsilon | 0.244 | review | 0 | prob=0.24; knownness=0.23 |
| 5 | bb_beta | 0.174 | review | 0 | prob=0.17; knownness=0.23 |

## Ana Projeden Farkı
Bu bağımsız proje genel plazmid önceliklendirme platformunu değil, tek bir biyolojik soruyu hedefler: plazmid omurgalarının coğrafi yayılım riskini erken saptamak.
