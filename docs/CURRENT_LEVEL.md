# Mevcut Seviye Değerlendirmesi

Tarih: 2026-05-26 (v2.0.0 final)
Checkpoint: `artifacts/cassiopeia_prime/cassiopeia_best.pt`
Eğitim sınırı: en fazla 2,048 plasmid, mevcut train split 1,414 plasmid

## Durum

Cassiopeia Prime v2.0.0 final: **589K parametre, en son benchmark v3.**

## Audited Metrics

`artifacts/cassiopeia_prime/benchmark_v3.json` çıktısı:

| Split | Mobility BA | AMR AUROC | Expansion AUROC | Task Score |
|---:|---:|---:|---:|---:|
| Validation | 76.26% | 91.66% | 83.57% | **83.83%** |
| Test | 70.28% | 89.94% | 87.03% | 82.42% |
| Held-out | 76.75% | 93.47% | 84.68% | **84.97%** |

## v1 vs v2.0.0 Karşılaştırması

| Metrik | v1 | v2.0.0 |
|---:|---:|---:|
| Validation Task Score | 83.66% | **83.83%** |
| Held-out Task Score | 85.78% | 84.97% |
| Parameters | 501K | **589K** |
| CV Task Score (5-fold) | — | **83.63%** |

## Teknik Notlar

- FRP pre-compute + eval_interval=3 ile eğitim ~3× hızlandı.
- SWA + L-BFGS temperature scaling + logistic regression calibration.
- 589,016 parametre, 122 test, %85 coverage, lint hatasız.
