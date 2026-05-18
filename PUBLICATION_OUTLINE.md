# Publication Outline

## Working Title

Multi-scale attention Transformers over sequence hashes for fast, DNA-only mobile genetic element risk prediction

## Core Claim

A lightweight, multi-scale attention Transformer over hashed sequence representations can predict AMR cargo and dissemination risk directly from raw DNA. It achieves state-of-the-art accuracy under strict no-metadata inference constraints while maintaining exact reverse-complement consistency, an ultra-low 1.40 MB footprint, and 24.7 ms/sequence latency.

## Abstract Skeleton

Mobile genetic elements drive antimicrobial resistance (AMR) spread, but predictive tools often rely on metadata or heavy annotation pipelines that leak context and restrict deployability. We introduce DNA Sentinel v2, a fast, DNA-only framework for mobile element risk analysis. DNA Sentinel v2 accept raw FASTA DNA and employs a lightweight KmerTransformer (<105k parameters) that reasons over hashed k-mer features at multiple bp scales (512, 2048, 8192 bp) with multi-head attention. On a strict group-aware split, KmerTransformer achieves an AMR cargo AUROC of 0.790 and plasmid dissemination AUROC of 0.883, outperforming both convolutional neural models and traditional sparse k-mer linear baselines. Operating at 24.7 ms/sequence on CPU with a 1.40 MB checkpoint size, the model provides exact reverse-complement prediction consistency and extracts interpretable evidence windows for downstream biological validation.

## Main Figures

1. Multi-scale KmerTransformer architecture: sequence hashes, random projections, and attention evidence pooling.
2. Leakage-aware dataset splits and duplicate controls.
3. Benchmark: KmerTransformer vs convolutional sparse-MIL vs hashed k-mer linear baselines.
4. Multitask prediction performance and ECE calibration.
5. Latency (24.7 ms/seq) and reverse-complement exact consistency audits.
6. Case study: annotation-blind evidence window localization for high-risk AMR plasmids.

## Key Experiments

- Strict group-aware split vs loose split leakage sweeps.
- Multi-scale window configuration sweeps (512, 2048, 8192 bp).
- Hashed k-mer feature dimension ablation (2048, 4096, 8192).
- Reverse-complement consensus alignment ablation.
- External holdout evaluation on independent genomic databases.
- Evidence-window validation against wet-lab HGT breakpoints.

## Honest Current Gap

The mobility prediction head remains challenging (0.594 balanced accuracy) under strict group-aware splitting, indicating that backbone mobility classes are heavily group-dependent. The strongest claim lies in direct, DNA-only AMR and high-dissemination expansion profiling with robust calibration.
