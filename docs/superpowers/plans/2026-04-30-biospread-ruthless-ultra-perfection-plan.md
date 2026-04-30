# BioSpread Ruthless Ultra-Perfection Plan (2026-04-30)

This plan is a "Level 10" extension of the previous excellence plans. It transitions BioSpread from a high-quality engineering prototype to a scientifically world-class early warning system.

## 1. Ruthless Critique Summary

### ML & Validation
*   **Arbitrary Labeling:** "2 countries in 3 years" ignores the difference between a hop across a border and a jump across an ocean.
*   **The "Substitution" Sin:** The current code occasionally uses OOF (Out-Of-Fold) metrics when temporal or external metrics are missing, which is a cardinal sin in high-stakes ML.
*   **Calibration Fragility:** Isotonic regression is used without enough data to justify its flexibility, leading to "steppy" and potentially overfit probability curves.
*   **Feature Attribution:** The current "attributions" (local explanations) are literally hardcoded to show the first 3 features. This is misleading and unacceptable for a "transparent" model.

### Biological Relevance
*   **AMR Naivety:** Treating all AMR genes as equal is biologically incorrect. One `blaNDM-1` (Carbapenemase) is worth more than ten `tet(A)` (Tetracycline resistance) genes in terms of public health risk.
*   **Host Range Flatness:** Counting host genera doesn't capture the difficulty of jumping across phylogenetic boundaries (e.g., Proteobacteria to Firmicutes).
*   **Static Mobility:** Mobility is treated as a point-in-time score rather than a potential (e.g., the presence of a "helper" plasmid nearby).

### Software Engineering
*   **Orchestration Bloat:** `pipeline.py` is a "God Object" that handles everything from raw file reading to HTML dashboard generation.
*   **Data Path Fragility:** The `auto` mode for input selection is convenient but dangerous for reproducibility.
*   **Missing Deep Tests:** 90% coverage is high, but the *branch* coverage for edge cases in the ensemble logic is insufficient.

---

## 2. Ultra-Perfection Implementation Roadmap

### Phase A: Scientific & Biological Sophistication
*   **[ ] Priority-Weighted AMR:** Create a `src/bio_spread_project/ontology.py` with a WHO-priority weight map for common AMR genes. Update `data.py` to use this.
*   **[ ] Phylogenetic Host Distance:** Implement a lookup for host genus to taxonomic order. Replace `n_hosts` with "Taxonomic Breadth" (number of unique orders).
*   **[ ] Spatial Connectivity Features:** Add a "Continent Count" and "Inter-continental Jump" flag to the feature set.
*   **[ ] Refined Labeling:** Add an optional "High Impact Spread" label: $\ge 2$ new macro-regions (continents/sub-continents) instead of just countries.

### Phase B: Advanced ML & Honest Interpretability
*   **[ ] Integration of Real SHAP:** Use `shap` library (or a fast lightweight approximation) for local explanations in `geo_reliability.py`.
*   **[ ] Probability Calibration Guardrails:** Add a check to switch from Isotonic to Platt Scaling when $N < 500$ to prevent overfitting.
*   **[ ] Uncertainty Bounds:** Implement a bootstrap-based "Risk Confidence Interval" (e.g., "Risk is 0.75 [0.68 - 0.82]").
*   **[ ] Strict Metric Separation:** Forbid *any* code path that substitutes OOF for temporal/external metrics in the release gate. If it's not there, it's `N/A`.

### Phase C: Architectural Hardening & Performance
*   **[ ] Decoupled Orchestration:** Split `pipeline.py` into `DataOrchestrator`, `ModelOrchestrator`, and `ArtifactManager`.
*   **[ ] Schema Enforcement:** Use a lightweight schema validator (or Polars native dtypes) to enforce strict data contracts at every stage.
*   **[ ] Parquet-Native Pipeline:** Transition the primary data format to Parquet for speed and schema stability, keeping CSV only for human-readable exports.
*   **[ ] Dependency Injection for I/O:** Make core ML logic accept Polars DataFrames/LazyFrames instead of file paths to enable 100% in-memory testing.

### Phase D: Judge-Ready Polish & Transparency
*   **[ ] Interactive Dashboard 2.0:** Replace the static HTML with a more interactive version (using a lightweight JS charting library or pre-rendered Plotly) showing:
    *   Calibration curves with confidence bands.
    *   Feature importance with SHAP summary plots.
    *   Geographic spread heatmaps (if lat/long available or via country mapping).
*   **[ ] Automated "Red-Teaming" Report:** Add a section to `audit.json` that intentionally tries to break the model (e.g., "Sensitivity to Year", "Sensitivity to Country Sampling").
*   **[ ] Scientific Limitations Document:** A dedicated `LIMITATIONS.md` that honestly describes where the model fails (e.g., "cannot predict spread of novel synthetic sequences not seen in RefSeq").

---

## 3. Execution Plan (Immediate Steps)

1.  **[ ] Audit Cleanup:** Fix the "attribution" fake-out in `geo_reliability.py` immediately.
2.  **[ ] Real Temporal Split:** Implement the 70/30 temporal split for the Geo model as described in the previous excellence plan.
3.  **[ ] Data Decoupling:** Start moving `read_table` and heuristic logic out of `data.py` into specialized loaders.
4.  **[ ] Priority AMR:** Add the first version of the AMR weight map.

This plan ensures that BioSpread is not just a "good project" but a "gold standard" for biological early warning ML.
