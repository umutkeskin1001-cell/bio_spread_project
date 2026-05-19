# BioSpread Experiment Debrief: DNA Sentinel Evolution

This document serves as a retrospective log of all architectural, optimization, and training strategies evaluated during the hardening of the **DNA Sentinel (KmerTransformer)** model. It outlines what worked, what failed, and the key lessons to prevent redundant exploration in the future.

---

## 1. What Worked (Successful Implementations)

### A. Core Architecture & Feature Engineering
*   **Vectorized Multi-Scale Window Extractor**: 
    *   *Implementation*: Replaced slow Python-loop string slice-and-count extraction with high-performance 2D Tensor `unfold` operations and flat-offset `bincount` histogram calculations.
    *   *Impact*: Feature preparation speed grew by **2.5x** (reduced from ~35s to 14s), enabling fully parallelized execution on CPU/MPS/CUDA.
*   **Native Attention & Learned Position Embeddings**:
    *   *Implementation*: Replaced custom, complex Circular RoPE attention blocks with PyTorch's native `nn.MultiheadAttention` and learned absolute positional embeddings (`nn.Parameter`).
    *   *Impact*: Standard attention is highly optimized for hardware accelerators (via PyTorch Inductor) and drastically simplified code readability while yielding superior generalization.
*   **Parallel Gated & Additive Input Fusion**:
    *   *Implementation*: Projects the high-dimensional (4096-dim) k-mer histograms and spectrograms through parallel gated (multiplicative `GELU`) and additive (`concatenation + linear projection`) paths.
    *   *Impact*: Enabled robust, multi-modal alignment of lexicographical and spectral dimensions.
*   **Stabilized Cross-Scale Interaction**:
    *   *Implementation*: Local features query macro-context through cross-attention, stabilized by explicit `LayerNorm` layers and a gated residual highway (`cross_gate`).
    *   *Impact*: Provided a strong inductive bias, helping local sequence details learn macro structure.
*   **Task-Routing Projections**:
    *   *Implementation*: Built bottleneck projections (`mob_proj`, `amr_proj`, `exp_proj`) to map pooled features to task-oriented manifolds before downstream MLP classification heads.
    *   *Impact*: Mitigated multi-task gradient conflicts by allowing heads to specialize their input manifolds.

### B. Loss & Training Strategies
*   **Task-Weighted Dual Focal Loss**:
    *   *Implementation*: Replaced standard binary cross-entropy with Focal Loss (`gamma=2.0`) coupled with positive weighting for AMR and Expansion prediction tasks.
    *   *Impact*: Successfully resolved structural sparsity and class imbalances, resulting in massive metric jumps.
*   **Dynamic Label-Smoothed Distillation**:
    *   *Implementation*: Leveraged temporal ensembling (EMA teacher) to distill soft, smoothed target predictions (`0.9 * p + 0.05`) to the active model.
    *   *Impact*: Prevented model overconfidence and acted as a powerful regularizer, lifting validation stability.

---

## 2. What Did NOT Work (Ineffective Attempts to Avoid)

### A. Self-Supervised Pre-training (MLM & Contrastive Alignment)
*   **The Trial**: Pre-trained the transformer model on the raw PLSDB sequences (~350MB of FASTA data) using Masked Language Modeling (k-mer reconstruction) and contrastive alignment (CLIP/PAWS-style strand-invariance).
*   **The Failure**: Downstream performance did not improve; in fact, training directly from scratch on consolidated features yielded much higher test scores.
*   **Why**: 
    1.  The sequence length was too short (28 windows total) and the model size too small to leverage high-capacity pre-training.
    2.  Contrastive strand-alignment before correcting consensus strand orientations forced the model to align misaligned complementary strands, leading to noisy representation collapse.

### B. Soft Mixture of Experts (Soft MoE) in Input Projections
*   **The Trial**: Built a routing network with 4 experts (`SoftMicroMoE`) to project 4096-dimensional k-mer histograms down to the hidden dimension.
*   **The Failure**: Caused severe overfitting and optimization instabilities.
*   **Why**: Slicing the representation path into expert parameters on small agricultural dataset samples created parameter bloat. A standard LayerNorm projection is vastly superior.

### C. Circular RoPE (Rotary Position Embeddings)
*   **The Trial**: Computed custom rotational cos/sin matrices to inject relative positional information into scale sequences.
*   **The Failure**: Yielded significantly lower downstream generalization.
*   **Why**: Relative positional context is less relevant than absolute scale context (e.g. knowing exactly which scale a window belongs to is absolute). Learned positional embeddings solved this cleanly.

### D. CLS Token Pooling
*   **The Trial**: Prepended a `[CLS]` token to sequence embeddings to act as the global representation accumulator.
*   **The Failure**: Dropped test Mobility balanced accuracy from 62.3% to 56.9%.
*   **Why**: In shallow networks (e.g., 2 layers), a single CLS token lacks the routing capacity to aggregate multi-scale features as efficiently as our custom multi-scale cross-attention pooling.

---

## 3. Final Metric Progression Summary

| Metric | Original Baseline | Native Attention + Consolidation | Final Hardened Model (Focal Loss + Routing) |
| :--- | :---: | :---: | :---: |
| **Mobility Balanced Accuracy** | 43.47% | 61.75% | **65.75%** (+22.28% vs Baseline) |
| **AMR AUROC** | 75.84% | 78.62% | **83.82%** (+7.98% vs Baseline) |
| **AMR AUPRC** | 67.64% | 74.14% | **80.16%** (+12.52% vs Baseline) |
| **Expansion AUROC** | 87.47% | 84.21% | **84.35%** |
| **Expansion AUPRC** | 70.69% | 71.51% | **71.31%** |
