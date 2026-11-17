"""Plot full training and validation loss histories.

Edit the settings below and run:

    uv run python examples/full/plot_loss.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis_tools.plotting import plot_loss, save_figure

# ============================================================
# Input and output
# ============================================================

WATCHMAL_ROOT = Path("path/to/WatChMaL")
RUN_DIRECTORY = Path("path/to/run")

OUTPUT_PATH = Path("outputs/loss/loss_curve.png")


# ============================================================
# Plot settings
# ============================================================

DPI = 200
SHOW_FIGURE = True

FIGSIZE = (15.0, 5.0)
ZOOM_PERCENTILES = (2.0, 98.0)

TRAINING_COLOR = "tab:blue"
VALIDATION_COLOR = "tab:orange"
BEST_COLOR = "crimson"


def main() -> None:
    """Load WatChMaL logs and plot loss histories."""
    sys.path.insert(0, str(WATCHMAL_ROOT))

    from analysis.read import WatChMaLOutput

    run = WatChMaLOutput(str(RUN_DIRECTORY))

    train_epoch = np.asarray(
        run.train_log_epoch,
        dtype=float,
    )
    train_loss = np.asarray(
        run.train_log_loss,
        dtype=float,
    )

    validation_epoch = np.asarray(
        run.val_log_epoch,
        dtype=float,
    )
    validation_loss = np.asarray(
        run.val_log_loss,
        dtype=float,
    )
    validation_best = np.asarray(
        run.val_log_best,
        dtype=bool,
    )

    positive_losses = np.concatenate(
        (
            train_loss[train_loss > 0.0],
            validation_loss[validation_loss > 0.0],
        )
    )

    if positive_losses.size == 0:
        raise ValueError(
            "No positive loss values were found."
        )

    epsilon = max(
        float(positive_losses.min()) * 0.1,
        1e-12,
    )

    train_loss_log = np.maximum(
        train_loss,
        epsilon,
    )
    validation_loss_log = np.maximum(
        validation_loss,
        epsilon,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=FIGSIZE,
        gridspec_kw={
            "width_ratios": [1.0, 1.35],
        },
    )
    full_axis, zoom_axis = axes

    # Full training history with a logarithmic y-axis.
    plot_loss(
        train_epoch,
        train_loss_log,
        ax=full_axis,
        xlabel="Epoch",
        ylabel="Loss",
        title="Full training history",
        yscale="log",
        label="Training loss",
        color=TRAINING_COLOR,
        linewidth=1.5,
        alpha=0.35,
    )

    plot_loss(
        validation_epoch,
        validation_loss_log,
        ax=full_axis,
        label="Validation loss",
        color=VALIDATION_COLOR,
        marker="o",
        linewidth=2.5,
        markersize=4.0,
    )

    full_axis.scatter(
        validation_epoch[validation_best],
        validation_loss_log[validation_best],
        color=BEST_COLOR,
        s=35,
        zorder=5,
        label="Best validation loss",
    )

    full_axis.grid(
        True,
        which="both",
        alpha=0.25,
    )
    full_axis.legend()

    # Linear-scale view of the central loss range.
    plot_loss(
        train_epoch,
        train_loss,
        ax=zoom_axis,
        xlabel="Epoch",
        ylabel="Loss",
        title="Zoomed loss",
        label="Training loss",
        color=TRAINING_COLOR,
        linewidth=1.5,
        alpha=0.45,
    )

    plot_loss(
        validation_epoch,
        validation_loss,
        ax=zoom_axis,
        label="Validation loss",
        color=VALIDATION_COLOR,
        marker="o",
        linewidth=2.5,
        markersize=5.0,
    )

    zoom_axis.scatter(
        validation_epoch[validation_best],
        validation_loss[validation_best],
        color=BEST_COLOR,
        s=40,
        zorder=5,
        label="Best validation loss",
    )

    zoom_minimum, zoom_maximum = np.percentile(
        positive_losses,
        ZOOM_PERCENTILES,
    )

    zoom_axis.set_ylim(
        max(float(zoom_minimum) * 0.5, 1e-8),
        float(zoom_maximum) * 1.2,
    )
    zoom_axis.grid(alpha=0.3)
    zoom_axis.legend()

    best_index = int(
        np.argmin(validation_loss)
    )

    print("\n========== Loss summary ==========")
    print(f"Run: {RUN_DIRECTORY}")
    print(
        "Best validation loss: "
        f"{validation_loss[best_index]:.6g}"
    )
    print(
        "Best epoch: "
        f"{validation_epoch[best_index]:.3f}"
    )
    print("==================================\n")

    figure.suptitle(
        "Training and validation loss",
        fontsize=16,
    )
    figure.tight_layout()

    save_figure(
        figure,
        OUTPUT_PATH,
        dpi=DPI,
        close=not SHOW_FIGURE,
    )

    print(f"Saved: {OUTPUT_PATH}")

    if SHOW_FIGURE:
        plt.show()


if __name__ == "__main__":
    main()