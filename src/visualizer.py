"""
Matplotlib Visualization Engine for Psychometric Profiles and Model Interpretability.
Employs standard Matplotlib with headless Agg backend.
"""

import io
import base64
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for servers and scripts
import matplotlib.pyplot as plt
import numpy as np

from .data_loader import SEVERITY_LEVELS


def create_radar_chart(
    dep_score: float,
    anx_score: float,
    str_score: float,
    as_base64: bool = True
) -> str:
    """
    Generates a 3-axis psychological triad radar/spider chart (Depression, Anxiety, Stress)
    normalized against maximum clinical ranges, and returns a base64 encoded PNG data URI.
    """
    categories = ["Depression", "Anxiety", "Stress"]
    # Normalize scores against clinical scale maxima (Dep: 42, Anx: 42, Str: 42)
    values = [
        min(1.0, dep_score / 42.0),
        min(1.0, anx_score / 42.0),
        min(1.0, str_score / 42.0),
    ]
    # Complete circular polygon
    values += values[:1]
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True), facecolor="#12161f")
    ax.set_facecolor("#181e29")

    # Grid line styling
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, color="#E2E8F0", size=11, weight="bold")

    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.50, 0.75, 1.0], ["Mild", "Mod", "Sev", "Ext"], color="#64748B", size=8)
    plt.ylim(0, 1.0)
    ax.spines["polar"].set_color("#334155")
    ax.grid(color="#334155", linestyle="--", linewidth=0.8)

    # Plot & fill area
    ax.plot(angles, values, color="#38BDF8", linewidth=2.5, linestyle="solid")
    ax.fill(angles, values, color="#38BDF8", alpha=0.35)

    # Markers for points
    ax.scatter(angles[:-1], values[:-1], color="#0284C7", s=60, zorder=5)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)

    if as_base64:
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    return buf.getvalue()


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    title: str = "Top Predictive Psychological Symptoms",
    top_n: int = 10,
    save_path: Optional[str] = None
):
    """Plots top features from Random Forest or Gradient Boosting using Matplotlib."""
    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices][::-1]
    top_scores = importances[indices][::-1]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#F8FAFC")
    ax.set_facecolor("#FFFFFF")
    bars = ax.barh(top_names, top_scores, color="#0EA5E9", edgecolor="#0284C7")

    ax.set_title(title, fontsize=13, fontweight="bold", color="#0F172A", pad=12)
    ax.set_xlabel("Relative Gini Importance", fontsize=10, color="#334155")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
