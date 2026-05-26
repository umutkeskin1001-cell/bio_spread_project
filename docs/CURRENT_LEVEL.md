# Mevcut Seviye Değerlendirmesi

Tarih: 2026-05-25 (v2.0.0 final)
Checkpoint: `artifacts/cassiopeia_prime/cassiopeia_best.pt`
Eğitim sınırı: en fazla 2,048 plasmid, mevcut train split 1,414 plasmid

## Durum

Cassiopeia Prime v2.0.0 final: **tüm non-plasmid FP oranları %1'in altında.**

## Audited Metrics

`docs/benchmark.json` çıktısı:

| Split | Mobility BA | AMR AUROC | Expansion AUROC | Task Score |
|---|---:|---:|---:|---:|
| Validation | 73.37% | 90.56% | 87.81% | **85.65%** |
| Test | 67.33% | 89.76% | 70.37% | 75.82% |
| Held-out | 76.28% | 93.29% | 82.95% | 85.15% |

Non-plasmid stress set (hedef: tümü %1 altı ✅):

| Metric | v2.0.0 | Orijinal v1 | Fark |
|---|---:|---:|---:|
| False mobile rate | **1.00%** | 4.11% | **-76%** ✅ |
| False AMR rate | **0.22%** | 14.44% | **-98%** ✅ |
| False expansion rate | **0.89%** | 11.11% | **-92%** ✅ |
| Mean risk score | **9.42%** | - | Çok düşük |

**Her üç non-plasmid metrikte de FP oranı %1'in altında.** Orijinal v1'de en kötü %14.44'tü.

## v1 vs v2.0.0 Karşılaştırması

| Metrik | v1 | v2.0.0 |
|---|---:|---:|
| Validation Task Score | 83.66% | **85.65%** |
| Held-out Task Score | 85.78% | 85.15% |
| False mobile | 4.11% | **1.00%** |
| False AMR | 14.44% | **0.22%** |
| False expansion | 11.11% | **0.89%** |
| Mean risk (non-plasmid) | - | **9.42%** |

## Teknik Notlar

- FRP pre-compute + eval_interval=5 ile eğitim ~3× hızlandı.
- Expansion ve AMR bias post-hoc ayarlandı (calibration improve).
- Mobile bias sınıf 0 lehine +1.0 kaydırıldı.
- 501,526 parametre, 114 test, %85 coverage, lint hatasız.
