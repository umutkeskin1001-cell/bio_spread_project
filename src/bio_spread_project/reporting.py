"""Markdown reporting for BioSpread."""

from __future__ import annotations

from typing import Any

from bio_spread_project.model import Prediction


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _fmt_metric(metrics: dict[str, float], key: str) -> str:
    if key not in metrics:
        return "not_evaluated"
    return _fmt(float(metrics[key]))


def _fmt_optional(value: object) -> str:
    if value is None:
        return "not_evaluated"
    if isinstance(value, (int, float)):
        return _fmt(float(value))
    return str(value)


def render_markdown_report(
    *,
    predictions: list[Prediction],
    metrics: dict[str, float],
    calibration: dict[str, Any],
    split_year: int,
    horizon_years: int,
    coefficient_summary: str = "",
) -> str:
    """Render a compact competition-friendly report."""
    top_rows = sorted(predictions, key=lambda row: row.risk_probability, reverse=True)[:10]
    primary_model = top_rows[0].model_name if top_rows else "unknown"
    lines = [
        "# BioSpread: Plazmid Coğrafi Yayılım Erken Uyarısı",
        "",
        "## Problem",
        (
            "Antimikrobiyal direnç taşıyan plazmid omurgaları farklı ülke ve "
            "konaklara yayıldığında halk sağlığı açısından takip edilmesi zor "
            "bir risk oluşur. Bu proje, geçmiş gözlemlerden yakın vadeli "
            "coğrafi yayılım riskini tahmin eder."
        ),
        "",
        "## Kurulum",
        f"Seçilen birincil model: `{primary_model}`.",
        f"Validation / doğrulama modu: `{metrics.get('validation_mode', 'direct')}`.",
        f"Eğitim gözlemleri: `{split_year}` ve öncesi.",
        f"Değerlendirme ufku: sonraki `{horizon_years}` yıl.",
        "",
        "## Validation And Reliability",
        f"- ROC AUC: `{_fmt(metrics['roc_auc'])}`",
        f"- Average precision: `{_fmt(metrics['average_precision'])}`",
        f"- Pozitif prevalans: `{_fmt(metrics['prevalence'])}`",
        f"- Top-k precision: `{_fmt(metrics['top_k_precision'])}`",
        f"- Abstain/review oranı: `{_fmt(metrics['abstain_rate'])}`",
        f"- Kalibrasyon hatası: `{_fmt(calibration['expected_calibration_error'])}`",
        f"- Brier score: `{_fmt(calibration['brier_score'])}`",
        f"- Group OOF ROC AUC: `{_fmt_metric(metrics, 'group_oof_roc_auc')}`",
        f"- Temporal holdout ROC AUC: `{_fmt_metric(metrics, 'temporal_holdout_roc_auc')}`",
        (
            "- Bootstrap ROC AUC CI: "
            f"`[{_fmt(metrics.get('bootstrap_roc_auc_ci_low', metrics['roc_auc']))}, "
            f"{_fmt(metrics.get('bootstrap_roc_auc_ci_high', metrics['roc_auc']))}]`"
        ),
        (
            "- Max single-feature AUC: "
            f"`{_fmt(metrics.get('max_single_feature_auc', 0.0))}` "
            f"(suspicious: `{int(float(metrics.get('suspicious_feature_count', 0.0)))}`)"
        ),
        f"- Leakage guard: `{metrics.get('leakage_audit_status', 'not_checked')}`",
        f"- Kalite kapıları: `{'pass' if metrics.get('all_quality_gates_passed') else 'review'}`",
        f"- Katsayı özeti: `{coefficient_summary or 'not_available'}`",
        "",
        "## Calibration",
        "| Bin | Mean prediction | Observed rate | Count |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in calibration.get("calibration_bins", []):
        lines.append(
            f"| {_fmt(float(row['bin_start']))}-{_fmt(float(row['bin_end']))} | "
            f"{_fmt_optional(row['mean_prediction'])} | {_fmt_optional(row['observed_rate'])} | "
            f"{int(float(row['count']))} |"
        )
    lines.extend(
        [
            "",
            "## Leakage And Audit",
            (
                "Feature columns are checked against future/outcome naming patterns, "
                "and single-feature AUC is monitored to catch near-deterministic leakage."
            ),
            "",
            "## Release Gate",
            (
                "Release readiness is determined from quality gates, drift checks, "
                "and model-registry trend evidence. Fresh output directories usually "
                "start as conditional_go until enough registry history exists."
            ),
            "",
            "## Limitations",
            (
                "This is a retrospective early-warning benchmark over packaged data. "
                "It is not clinical diagnosis, a patient-level decision system, or proof "
                "of field deployment performance."
            ),
            "",
            "## Reproducibility",
            (
                "The run writes input hashes, selected input mode, threshold sources, "
                "environment versions, model registry entries, and release-gate artifacts."
            ),
            "",
            "## En Riskli Adaylar",
            "| Sıra | Backbone | Risk | Güven | Yeni ülke | Açıklama |",
            "| --- | --- | ---: | --- | ---: | --- |",
        ]
    )
    for rank, row in enumerate(top_rows, start=1):
        lines.append(
            "| "
            f"{rank} | {row.backbone_id} | {_fmt(row.risk_probability)} | "
            f"{row.confidence_tier} | {row.n_new_countries_future} | {row.explanation} |"
        )
    lines.extend(
        [
            "",
            "## Ana Projeden Farkı",
            (
                "Bu bağımsız proje genel plazmid önceliklendirme platformunu değil, "
                "tek bir biyolojik soruyu hedefler: plazmid omurgalarının coğrafi "
                "yayılım riskini erken saptamak."
            ),
            "",
        ]
    )
    return "\n".join(lines)
