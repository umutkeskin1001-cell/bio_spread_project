from __future__ import annotations

import json
import os
from pathlib import Path

from bio_spread_project.orchestrator import run_pipeline


BASE_FLAGS_PATH = Path("project_config/config/enriched_features.yaml")
GEO_INPUT = Path("reports/latest_refresh/features.csv")
OUTPUT_ROOT = Path("reports/ablation_matrix_v2")


def _read_flags() -> dict[str, bool]:
    raw = BASE_FLAGS_PATH.read_text(encoding="utf-8").splitlines()
    out: dict[str, bool] = {}
    for line in raw:
        clean = line.strip()
        if not clean or clean.startswith("#") or ":" not in clean:
            continue
        key, value = clean.split(":", 1)
        out[key.strip()] = value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return out


def _write_flags(path: Path, flags: dict[str, bool]) -> None:
    lines = [f"{k}: {'true' if v else 'false'}" for k, v in sorted(flags.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_ablation() -> Path:
    base = _read_flags()
    scenarios: list[tuple[str, dict[str, bool]]] = [("all_on", dict(base))]

    for key in (
        "enable_synergy_interactions",
        "enable_phylo_propagation",
        "enable_evidential_meta",
        "enable_grps",
        "enable_phylo_spatial_embedding",
        "enable_conformal",
    ):
        variant = dict(base)
        variant[key] = False
        scenarios.append((f"no_{key}", variant))

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, float | str]] = {}
    temp_flags = OUTPUT_ROOT / "tmp_enriched_flags.yaml"
    for name, flags in scenarios:
        _write_flags(temp_flags, flags)
        env_prev = os.environ.get("BIO_SPREAD_ENRICH_FLAGS_PATH")
        os.environ["BIO_SPREAD_ENRICH_FLAGS_PATH"] = str(temp_flags)
        try:
            run = run_pipeline(
                run_mode="geo",
                geo_spread_features_path=GEO_INPUT,
                output_dir=OUTPUT_ROOT / name,
            )
        finally:
            if env_prev is None:
                os.environ.pop("BIO_SPREAD_ENRICH_FLAGS_PATH", None)
            else:
                os.environ["BIO_SPREAD_ENRICH_FLAGS_PATH"] = env_prev
        summary[name] = {
            "roc_auc": float(run.metrics.get("roc_auc", 0.0)),
            "average_precision": float(run.metrics.get("average_precision", 0.0)),
            "temporal_holdout_roc_auc": float(run.metrics.get("temporal_holdout_roc_auc", 0.0)),
            "temporal_consistency_status": str(run.metrics.get("temporal_consistency_status", "not_evaluated")),
            "feature_lineage_status": str(run.metrics.get("feature_lineage_status", "unknown")),
        }

    summary_path = OUTPUT_ROOT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    return summary_path


if __name__ == "__main__":
    path = run_ablation()
    print(path)
