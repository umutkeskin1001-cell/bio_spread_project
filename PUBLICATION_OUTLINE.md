# Publication Outline

## Working Title

DNA-only sparse evidence models for low-compute mobile genetic element risk prediction

## Core Claim

A compact sequence-only model can predict high-dissemination plasmid risk from raw DNA under strict no-metadata inference constraints, while preserving reverse-complement consistency and local evidence windows.

## Abstract Skeleton

Mobile genetic elements drive antimicrobial resistance spread, but many predictive systems rely on metadata, taxonomy, geography, or annotation pipelines that leak context and reduce deployability. We introduce DNA Sentinel, a low-compute DNA-only framework for mobile element risk analysis. DNA Sentinel accepts raw FASTA sequences and predicts mobility, AMR cargo, and high-dissemination risk using reverse-complement-consensus sequence models. On a strict group-aware plasmid split, the final model achieves high-dissemination AUROC 0.845 and AUPRC 0.753 with a 2.63 MB checkpoint. We show that a simple hashed k-mer model outperforms a neural sparse-MIL model in the current low-data regime, emphasizing benchmark discipline over architectural novelty. The framework includes leakage auditing, calibration reporting, and evidence-window extraction for downstream biological validation.

## Main Figures

1. System overview: raw DNA to risk and evidence windows.
2. Leakage-aware dataset construction.
3. Neural sparse-MIL vs k-mer benchmark.
4. Test performance and calibration.
5. Reverse-complement consistency and latency.
6. Example evidence windows for high-risk plasmids.

## Key Experiments

- Strict group-aware split vs loose split.
- Train-size sweep: 512, 2048, 4096.
- RC consensus ablation.
- Hashed k-mer ranges: 4-6, 5-6, 5-7.
- External holdout on independent plasmid collection.
- Annotation-blind evidence-window validation after inference.

## Honest Current Gap

The mobility head is not strong enough for a top-tier standalone claim. The most publishable current result is high-dissemination DNA-only prediction with strict leakage control.
