import matplotlib.pyplot as plt
import numpy as np


def plot_residual_histogram(predictions, targets, *, bins=50, title="Residuals"):
    """Plot a residual histogram and return the unclosed figure."""
    residuals = np.asarray(predictions) - np.asarray(targets)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(residuals, bins=bins, histtype="step")
    ax.axvline(0.0, color="black", linestyle="--")
    ax.set(xlabel="Prediction - target", ylabel="Events", title=title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
