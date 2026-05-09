"""Visualization engine for BioSpread reports."""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


def save_table_as_png(headers: list[str], rows: list[list[Any]], title: str, output_path: Path) -> None:
    """Render a table as a chic PNG image."""
    # Use a clean font if available, else default
    plt.rcParams['font.family'] = 'sans-serif'

    fig, ax = plt.subplots(figsize=(14, len(rows) * 0.5 + 1.5))
    ax.axis('off')

    # Custom styling
    colors = [['#ffffff' for _ in range(len(headers))] for _ in range(len(rows))]
    for i in range(len(rows)):
        if i % 2 == 0:
            for j in range(len(headers)):
                colors[i][j] = '#f1f3f5'

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellColours=colors,
        loc='center',
        cellLoc='center',
        colColours=['#343a40'] * len(headers),
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.2)

    # Style headers and cells
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.get_text().set_color('white')
            cell.get_text().set_fontweight('bold')
            cell.set_facecolor('#343a40')
        cell.set_edgecolor('#dee2e6')
        cell.set_linewidth(0.5)

    plt.title(title, fontsize=18, pad=30, weight='bold', color='#212529')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_summary(metrics: dict[str, Any], output_path: Path) -> None:
    """Generate a chic performance summary graphic."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#f8f9fa')

    # 1. Feature Importance
    top_features = metrics.get("top_features", [])
    if top_features:
        names = [f["feature"] for f in top_features[:10]][::-1]
        scores = [f["score"] for f in top_features[:10]][::-1]

        ax1.barh(names, scores, color='#339af0', edgecolor='white')
        ax1.set_title("TOP 10 PREDICTIVE SIGNALS", weight='bold', size=14, pad=15)
        ax1.grid(axis='x', linestyle='--', alpha=0.4)
        ax1.set_facecolor('white')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
    else:
        ax1.text(0.5, 0.5, "Signal Data N/A", ha='center', size=14)

    # 2. Key Metrics Bar Chart
    keys = ["roc_auc", "average_precision", "group_oof_roc_auc", "temporal_holdout_roc_auc"]
    labels = ["ROC AUC", "Avg Prec", "Group OOF", "Temporal"]
    values = [metrics.get(k, 0.0) for k in keys]

    colors = ['#22b8cf', '#20c997', '#94d82d', '#fcc419']
    ax2.bar(labels, values, color=colors, width=0.6)
    ax2.set_ylim(0, 1.0)
    ax2.set_title("CROSS-VALIDATION RIGOR", weight='bold', size=14, pad=15)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.set_facecolor('white')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    for i, v in enumerate(values):
        ax2.text(i, v + 0.02, f"{v:.3f}", ha='center', weight='bold')

    plt.tight_layout(pad=4.0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
