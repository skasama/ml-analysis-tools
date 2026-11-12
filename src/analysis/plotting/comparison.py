import matplotlib.pyplot as plt
import numpy as np


def plot_metric_comparison(metrics, *, metric, title=None):
    """Plot one scalar metric for each method."""
    methods = list(metrics)
    values = [metrics[method][metric] for method in methods]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(methods, values)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} by method")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_residual_comparison(results, *, bins=60, title="Residual comparison"):
    """Overlay residual distributions for two or more MethodResult objects."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for result in results:
        residuals = np.asarray(result.predictions) - np.asarray(result.targets)
        ax.hist(
            residuals.ravel(),
            bins=bins,
            histtype="step",
            linewidth=1.5,
            label=result.method,
        )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Prediction - target", ylabel="Events", title=title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig
