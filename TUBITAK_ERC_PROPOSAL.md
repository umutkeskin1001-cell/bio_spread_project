# TÜBİTAK / ERC Proposal Draft

## Project Title

DNA-only early warning models for mobile antimicrobial resistance dissemination

## Vision

Develop a low-compute, metadata-free AI system that screens plasmid DNA sequences for dissemination risk and highlights candidate sequence regions for downstream biological validation.

## Scientific Problem

AMR surveillance often depends on metadata and annotation pipelines that are incomplete, biased, or unavailable in early outbreak contexts. A DNA-only risk model would be deployable on raw sequence submissions and robust to missing epidemiological context.

## Hypothesis

Mobile AMR dissemination leaves detectable nucleotide-level signatures in plasmid sequence organization. These signatures can be learned with compact, sample-efficient models when leakage is controlled and reverse-complement symmetry is built into inference.

## Objectives

1. Build a curated, leakage-audited plasmid DNA benchmark.
2. Develop DNA-only models for AMR cargo, mobility, and high-dissemination risk.
3. Create explanation methods that identify candidate risk windows.
4. Validate evidence windows against external annotations after inference.
5. Deliver a lightweight CLI/API tool for surveillance teams.

## Methodology

- Strict FASTA-only model input.
- Offline labels from curated plasmid tables.
- Group-aware and near-duplicate-aware split construction.
- Reverse-complement-consensus prediction.
- Low-compute hashed k-mer and sparse-MIL neural baselines.
- Calibration, Brier score, ECE, and leakage audits as mandatory metrics.

## Work Packages

| WP | Scope | Output |
|---|---|---|
| WP1 | Data curation and leakage audit | Reproducible benchmark |
| WP2 | DNA-only model development | CLI/API + checkpoints |
| WP3 | Evidence-window validation | Biological interpretation report |
| WP4 | External validation | Independent holdout metrics |
| WP5 | Deployment and documentation | Dockerized production package |

## Risk Management

- If mobility remains weak, focus publication claim on high-dissemination and AMR cargo.
- If expansion labels show sampling bias, replace country breadth with external dissemination labels.
- If neural models underperform, keep the simpler k-mer model as the production path.

## Impact

The project enables low-cost sequence-only AMR risk triage without requiring sensitive metadata, large GPUs, or annotation pipelines.
