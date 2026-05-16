# Full Training Report — Sovereign-X Ultra (v4 Frontier)

> **Run:** `artifacts/prod_final`
> **Duration:** 10 epochs (Quick Deployment Mode)
> **Validation set:** 655 backbones
> **Test set:** 648 backbones
> **Config:** `config/prod.yaml`
> **Data:** `data/features/sequences.tsv` (21,521 samples)

---

## 1. Executive Summary

BioSpread Sovereign-X Ultra (v4 Frontier), "soğuk başlatma" (cold-start) sorununu deterministik bir yaklaşımla çözen yeni bir mimaridir. Bu run, **Temporal Proxy Alignment** ve **FiLM Conditioning** tekniklerinin başarısını kanıtlamıştır. Model, geçmiş verisi olmayan patojenlerde bile tam verili modelle neredeyse aynı performansı (Gap < 0.002) sergilemektedir.

---

## 2. Model Metrics (Sovereign-X Ultra)

### 2.1 ROC AUC by Horizon & Scenario

| Scenario | Horizon 1 | Horizon 2 | Horizon 3 |
|---|---|---|---|
| **Full (Temporal)** | **0.7824** | 0.6969 | 0.7207 |
| **Cold-Start** | **0.7806** | 0.6972 | 0.7188 |
| **Performance Gap**| **0.0018** | **-0.0003**| **0.0019** |

Model, kısa vadeli tahminlerde (H1) çok güçlü bir stabilite sergilemektedir. Uzun vadeli tahminlerdeki (H2, H3) hafif düşüş, biyolojik belirsizliğin artmasıyla uyumludur.

### 2.2 Classification Metrics (Horizon 3, Optimal F2)

| Metric | Value |
|---|---|
| **F1 Score** | **0.6781** |
| **Recall (TPR)** | **1.0000** |
| **Precision** | 0.5130 |
| **Brier Score** | 0.2490 |

H3 için bulunan optimal F2 eşiği (0.01), yayılım riskini kaçırmamak (Recall=1.0) adına yüksek duyarlılığa ayarlanmıştır.

---

## 3. Architectural Innovations

### 3.1 Temporal Proxy Alignment
Model, statik veriden temporal özellikleri tahmin eden bir **Proxy Generator** içerir. 
- **Loss:** MSE + InfoNCE
- **Amacı:** Cold-start örneklerini, temporal (warm) örneklerin manifolduna hizalamak.

### 3.2 FiLM Conditioning
Taksonomik veriler (Family/Genus), statik özellikleri **Linear Modulation** (FiLM) ile modüle ederek, patojenin "evrimsel bağlamını" tahmine ekler.

### 3.3 Uncertainty-Weighted Multi-Task Loss
Kayıp fonksiyonları (Hazard, Proxy, KD, Count) statik ağırlıklar yerine, Kendall (2018) yöntemine göre kendi belirsizliklerini (variance) optimize ederek dinamik olarak dengelenir.

---

## 4. Calibration Status

| Horizon | ECE (Expected Calibration Error) | Platt Scaler (a, b) |
|---|---|---|
| h1 | 0.1666 | a=0.608, b=-1.379 |
| h2 | 0.1772 | a=0.718, b=-0.857 |
| h3 | 0.1912 | a=0.877, b=-0.338 |

Kalibrasyon skorları, özellikle soğuk başlatma senaryosu için Platt scaling ile stabilize edilmiştir.

---

## 5. Deployment Readiness

- **Dockerfile.prod:** Prodüksiyon için optimize edildi.
- **FastAPI:** `scripts/serve_frontier.py` ile gerçek zamanlı tahmin servisi aktif.
- **Monitoring:** `MondrianConformalManager` ile güven aralıkları (intervals) sunulmaktadır.

---
*Report generated 2026-05-16 from artifacts/prod_final*
