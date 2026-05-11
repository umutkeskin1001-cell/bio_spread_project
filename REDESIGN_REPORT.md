# BioSpread Architectural Redesign Report

## Current Status: PHASE 1 COMPLETE
The project has undergone a complete architectural overhaul to eliminate data leakage and modernize the predictive pipeline.

### 1. Architecture Overhaul
- **Temporal Snapshot Builder**: Implemented to ensure zero future leakage. Labels and features are strictly computed based on historical snapshots.
- **GeneEncoder (Transformer)**: Replaced naive MLP/Bag-of-Words with a sequence-aware Transformer encoder.
- **TimeGate (Fourier)**: Replaced sigmoid-gated linear time projection with learnable Fourier features (`sin`/`cos`) for better temporal dynamics.
- **Evidential Learning**: Maintained the Dirichlet-based evidential head for uncertainty estimation.

### 2. Verified Metrics (Leakage-Free)
After removing the "cheating" mechanisms (temporal corruption), the current metrics on `snapshot_records.tsv` are:
- **ROC AUC**: 0.5858
- **Uncertainty AUC**: 0.5741
- **Recall**: 1.0000 (Model currently biased towards positive class)
- **Status**: Early stopping triggered at Epoch 6.

### 3. Critical Observations
- **Data Hardening**: The model is no longer "memorizing" future aggregates. The task is significantly harder but scientifically valid.
- **Feature Sparsity**: The current model only uses genetic sequences and years. It lacks the complex ecological and phenotypic features present in the original (leaky) dataset.
- **Performance Gap**: There is a significant gap between the "fake" 0.90+ AUC and the "real" 0.58 AUC. Bridging this gap requires a "genius-level" strategy.

### 4. Technical Debt Resolved
- CLI fixed (outputs saved).
- Configuration validated with Pydantic.
- Artifact management versioned.
- CI/Makefile aligned.

---
*Report generated for DeepSeek review.*
