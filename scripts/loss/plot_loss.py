"""Plot training and validation loss from a WatChMaL run."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis_tools.io import load_yaml
from analysis_tools.plotting import plot_loss, save_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot training and validation loss.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the loss configuration YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    watchmal_root = Path(config["watchmal_root"]).expanduser()
    run_directory = Path(config["run_directory"]).expanduser()
    output_path = Path(config["output"]["path"]).expanduser()

    sys.path.insert(0, str(watchmal_root))

    from analysis.read import WatChMaLOutput

    run = WatChMaLOutput(str(run_directory))

    train_epoch = np.asarray(run.train_log_epoch, dtype=float)
    train_loss = np.asarray(run.train_log_loss, dtype=float)

    validation_epoch = np.asarray(run.val_log_epoch, dtype=float)
    validation_loss = np.asarray(run.val_log_loss, dtype=float)
    validation_best = np.asarray(run.val_log_best, dtype=bool)

    positive_losses = np.concatenate(
        (
            train_loss[train_loss > 0.0],
            validation_loss[validation_loss > 0.0],
        )
    )

    if positive_losses.size == 0:
        raise ValueError("No positive loss values were found.")

    epsilon = max(float(positive_losses.min()) * 0.1, 1e-12)

    train_loss_log = np.maximum(train_loss, epsilon)
    validation_loss_log = np.maximum(validation_loss, epsilon)

    plot_config = config["plot"]
    training_style = plot_config["training"]
    validation_style = plot_config["validation"]
    best_style = plot_config["best"]

    figure, axes = plt.subplots(
        1,
        2,
        figsize=plot_config["figsize"],
        gridspec_kw={"width_ratios": [1.0, 1.35]},
    )
    full_axis, zoom_axis = axes

    plot_loss(
        train_epoch,
        train_loss_log,
        ax=full_axis,
        xlabel="Epoch",
        ylabel="Loss",
        title="Full training history",
        yscale="log",
        label="Training loss",
        **training_style,
    )

    plot_loss(
        validation_epoch,
        validation_loss_log,
        ax=full_axis,
        label="Validation loss",
        **validation_style,
    )

    full_axis.scatter(
        validation_epoch[validation_best],
        validation_loss_log[validation_best],
        color=best_style["color"],
        s=best_style["size"],
        zorder=5,
        label="Best validation loss",
    )

    full_axis.grid(
        True,
        which="both",
        alpha=0.25,
    )
    full_axis.legend()

    plot_loss(
        train_epoch,
        train_loss,
        ax=zoom_axis,
        xlabel="Epoch",
        ylabel="Loss",
        title="Zoomed loss",
        label="Training loss",
        **training_style,
    )

    plot_loss(
        validation_epoch,
        validation_loss,
        ax=zoom_axis,
        label="Validation loss",
        **validation_style,
    )

    zoom_axis.scatter(
        validation_epoch[validation_best],
        validation_loss[validation_best],
        color=best_style["color"],
        s=best_style["size"],
        zorder=5,
        label="Best validation loss",
    )

    zoom_min, zoom_max = np.percentile(
        positive_losses,
        plot_config["zoom_percentiles"],
    )

    zoom_axis.set_ylim(
        max(float(zoom_min) * 0.5, 1e-8),
        float(zoom_max) * 1.2,
    )
    zoom_axis.grid(True, alpha=0.3)
    zoom_axis.legend()

    best_index = int(np.argmin(validation_loss))

    print(f"Run: {config['name']}")
    print(f"Best validation loss: {validation_loss[best_index]:.6g}")
    print(f"Best epoch: {validation_epoch[best_index]:.3f}")

    figure.suptitle("Training and validation loss")
    figure.tight_layout()

    save_figure(
        figure,
        output_path,
        dpi=config["output"]["dpi"],
        close=True,
    )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
