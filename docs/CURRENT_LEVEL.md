# Mevcut Seviye Değerlendirmesi

Tarih: 2026-06-05 (Cassiopeia Prime v0.3.0, şampiyon: v14+v15 ensemble)
Checkpoints: `artifacts/cassiopeia_prime_v15/cassiopeia_best.pt` + `artifacts/cassiopeia_prime_v14/cassiopeia_best.pt`
Ensemble weights: v15 0.47 / v14 0.53
Eğitim sınırı: en fazla 2,048 plasmid, mevcut train split 1,466 plasmid

## Durum

Cassiopeia Prime v0.3.0: **568,437 parametre, 5.81 MB × 2 checkpoint, 375 test, ~%89 coverage, lint hatasız.**

## Audited Metrics (Ensemble)

`artifacts/cassiopeia_prime_v15/report_ensemble.json` çıktısı:

| Split | Mobility BA | AMR AUROC | Expansion AUROC | Task Score |
|---:|---:|---:|---:|---:|
| Validation | 77.63% | 91.99% | 88.54% | 86.05% |
| Test | 74.18% | 92.16% | 87.43% | 84.59% |
| Held-out | **78.33%** | **93.89%** | **88.78%** | **87.00%** |

### Held-out Mobility Per-class F1

| Sınıf | Precision | Recall | F1 |
|---:|---:|---:|---:|
| non-mobilizable (0) | 0.770 | 0.715 | 0.741 |
| **mobilizable (1)** | **0.695** | **0.756** | **0.725** |
| conjugative (2) | 0.877 | 0.878 | 0.878 |

## v14 → v15 → Ensemble İyileştirmeleri

| Metrik | v14 | v15 | Ensemble | Δ (v14→Ens) |
|---:|---:|---:|---:|---:|
| Held-out Task Score | 85.95% | 85.86% | **87.00%** | **+1.05%** |
| Mobility BA | 76.94% | 76.94% | **78.33%** | **+1.39%** |
| AMR AUROC | 93.91% | 93.20% | **93.89%** | -0.02% |
| Expansion AUROC | 87.01% | 87.44% | **88.78%** | **+1.77%** |
| Class 1 F1 | 0.678 | 0.716 | **0.725** | **+0.047** |
| Class 1 Recall | 64.7% | 73.1% | **75.6%** | **+10.9pp** |

## Teknik Notlar

- Mobilizable F1 iyileştirmesi: per-class weight + mobility-only focal loss (γ=0.5) + balanced sampling'de mobility 2× ağırlıklı.
- Ensemble: v14 (weight 0.47) + v15 (weight 0.53) probability averaging, tüm task score'u 87.00%'e taşıdı.
- FRP pre-compute + eval_interval=2 ile eğitim hızlandı.
- L-BFGS temperature + bias scaling ile kalibrasyon; her modele ayrı uygulanır, sonra ensemble ortalaması alınır.
- `dna` kısa CLI alias grubu (`dna train`, `dna predict`, `dna bench`, ...) eklendi.
- Biyolojik yorumlama (güven etiketi, CARD ailesi, mobility+AMR birlikte değerlendirmesi) eklendi.
- Web arayüzü (Predict + Benchmark) tamamen offline çalışır.
- Ensemble benchmark CLI'da `--ensemble-checkpoint / -e` ve `--ensemble-weight` parametreleri ile desteklenir.
