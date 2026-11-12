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


def plot_bias_resolution_comparison(metrics, *, title="Bias and resolution comparison"):
    """Plot bias and 68% resolution for every method in one figure."""
    methods = list(metrics)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, metric, label in (
        (axes[0], "bias", "Bias"),
        (axes[1], "resolution_68", "68% resolution"),
    ):
        values = [metrics[method][metric] for method in methods]
        axis.bar(methods, values)
        axis.set_ylabel(label)
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.3)
        axis.tick_params(axis="x", rotation=20)
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_classification_comparison(
    metrics,
    *,
    title="Classification efficiency, purity, and mis-PID",
):
    """Plot class-wise classification metrics for every method."""
    labels = sorted(
        {label for values in metrics.values() for label in values["labels"]}
    )
    methods = list(metrics)
    figure, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    width = 0.8 / len(methods)
    positions = np.arange(len(labels))
    for axis, metric, label in zip(
        axes,
        ("efficiency", "purity", "mis_pid"),
        ("Efficiency", "Purity", "Mis-PID rate"),
        strict=True,
    ):
        for method_index, method in enumerate(methods):
            values = dict(
                zip(metrics[method]["labels"], metrics[method][metric], strict=True)
            )
            heights = [values.get(class_label, np.nan) for class_label in labels]
            axis.bar(
                positions + method_index * width,
                heights,
                width=width,
                label=method,
            )
        axis.set(
            title=label,
            xlabel="Class",
            xticks=positions + width * (len(methods) - 1) / 2,
            xticklabels=labels,
            ylim=(0.0, 1.0),
        )
        axis.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Fraction")
    axes[-1].legend()
    figure.suptitle(title)
    figure.tight_layout()
    return figure


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
