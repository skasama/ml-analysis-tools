import matplotlib.pyplot as plt


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
