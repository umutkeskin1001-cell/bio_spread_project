# BioSpread Phase 3: Plasmid Intelligence Unit (PIU) Redesign Report

## 🎯 Executive Summary
The BioSpread surveillance pipeline has been successfully upgraded to the **PIU Phase 3 Architecture**. This redesign focused on transforming the model from a simple genetic classifier into a multi-modal evidential learning system that integrates genetic, functional, and epidemiological signals while strictly eliminating historical data leakage.

**Performance Achieved:** Authenticated Val ROC AUC **0.6037** (Leakage-free).

---

## 🛠️ Key Technical Implementations

### 1. Multi-Modal FusionNetV3
- **Genetic Stream:** Transformer-based GeneEncoder for raw plasmid sequences.
- **Functional Stream:** Multi-hot encoding of biological functions (AMR, Mobility, Metabolism).
- **Epidemiological Stream:** `TemporalContextEncoder` for historical spread tracking.
- **Pillar Feature Fuser:** Integration of **PFP** (Protein Fitness), **EBV** (Epistatic Pressure), and **CAV** (Geographic Aura).

### 2. Leakage Sterilization
- **Snapshot-Aware CAV:** Geographic aura vectors are now computed dynamically for each snapshot, ensuring the model only sees countries visited *up to* that specific year.
- **Logarithmic Scaling:** Applied `log1p` transformation to skewed numeric features (e.g., records counts), preventing feature dominance and stabilizing gradients.
- **Temporal Split:** Strict cutoff at Year 2020 for the training set, with all auxiliary features (like Epistasis) built strictly on past data.

### 3. Evidential Ranking Loss
- **Dirichlet Uncertainty:** Model outputs both spread probability and an explicit uncertainty score.
- **Paired Sampling:** Re-engineered the data loader to provide explicit snapshot pairs of the same backbone, allowing **RankNet Loss** to optimize the risk-ranking gradient at every step.

---

## 📈 Results & Observations
The model reached a stable AUC of ~0.60. While this is a significant baseline, it indicates a plateau where genetic profile alone might not fully explain spread variance without deeper host-context or higher-order epistasis modeling.

---

## 🚀 Next Steps (Phase 4)
- **Host-Niche Integration:** Incorporate host genus/species distributions into the Geographic Aura.
- **Sequence Augmentation:** Use ESM-2 embeddings directly in the GeneEncoder.
- **Soft-Targeting:** Transition from binary spread labels to continuous "Spread Intensity" forecasting.
