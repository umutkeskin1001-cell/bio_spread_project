# BioSpread 3.0 Core - Surveillance Report

## 📊 Model Performance Metrics
- **ROC AUC:** 0.8437
- **Average Precision:** 0.8073
- **Expected Calibration Error (ECE):** 0.2481
- **GNN Trigger Rate (Compute Efficiency):** 0.0%

## 🛡️ Top Threat Candidate Registry
| backbone_id   |   label_geo_spread |   risk_score | gnn_triggered   |   uncertainty |
|:--------------|-------------------:|-------------:|:----------------|--------------:|
| AA176         |                  1 |     0.948644 | False           |     0.0941555 |
| AA319         |                  1 |     0.923732 | False           |     0.121666  |
| AB187         |                  1 |     0.915611 | False           |     0.154054  |
| AA598         |                  1 |     0.908974 | False           |     0.167275  |
| AB685         |                  1 |     0.907969 | False           |     0.17705   |
| AA614         |                  1 |     0.903301 | False           |     0.131316  |
| AB710         |                  0 |     0.9022   | False           |     0.156417  |
| AA921         |                  1 |     0.896439 | False           |     0.105773  |
| AB745         |                  1 |     0.894538 | False           |     0.180257  |
| AC122         |                  1 |     0.89215  | False           |     0.141309  |
| AA083         |                  1 |     0.889546 | False           |     0.211165  |
| AA144         |                  1 |     0.880618 | False           |     0.179717  |
| AA619         |                  1 |     0.877537 | False           |     0.203166  |
| AC509         |                  1 |     0.87048  | False           |     0.211269  |
| AF042         |                  0 |     0.855575 | False           |     0.288772  |

## 🧪 Architecture Integrity
- **Dual-Uncertainty:** Enabled (EDL + MC Dropout)
- **Temporal Decay:** $\lambda = 0.3$
- **LightGCN Support:** Active (Degree-Penalized)
