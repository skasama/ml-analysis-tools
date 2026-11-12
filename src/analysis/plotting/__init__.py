from pathlib import Path

import matplotlib.pyplot as plt

from .comparison import (
    plot_bias_resolution_comparison,
    plot_metric_comparison,
    plot_residual_comparison,
)
from .diagnostics import plot_residual_histogram


def scatter_identity(
    predictions,
    targets,
    *,
    xlabel,
    ylabel,
    title,
    unit="",
    point_size=5,
    alpha=0.25,
):
    """Create a prediction-versus-truth scatter plot with an identity line."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(targets, predictions, s=point_size, alpha=alpha, rasterized=True)
    value_min = min(float(min(targets)), float(min(predictions)))
    value_max = max(float(max(targets)), float(max(predictions)))
    ax.plot(
        [value_min, value_max],
        [value_min, value_max],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Perfect prediction",
    )
    ax.set(xlabel=f"{xlabel} [{unit}]", ylabel=f"{ylabel} [{unit}]", title=title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def save(fig, path, dpi=200):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


__all__ = [
    "plot_metric_comparison",
    "plot_bias_resolution_comparison",
    "plot_residual_comparison",
    "plot_residual_histogram",
    "save",
    "scatter_identity",
]
