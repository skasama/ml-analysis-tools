import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.paths import input_path, output_path
from analysis.plotting import save


def column(data, names):
    avail = data.dtype.names or ()
    norm = {x.lower().replace("_", ""): x for x in avail}
    for x in names:
        if x.lower().replace("_", "") in norm:
            return np.asarray(data[norm[x.lower().replace("_", "")]], float)
    raise KeyError(f"Columns {names} not found. Available: {list(avail)}")


def plot_validation_loss(
    ax,
    epoch,
    loss,
    *,
    use_thin_line,
    label="Validation loss",
):
    if use_thin_line:
        ax.plot(
            epoch,
            loss,
            color="tab:orange",
            linewidth=0.5,
            alpha=0.3,
            label=label,
        )
    else:
        ax.plot(
            epoch,
            loss,
            color="tab:orange",
            linewidth=1.5,
            marker="o",
            markersize=4,
            label=label,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--train-log",
        type=Path,
        default=Path("outputs/log_train_0.csv"),
    )
    p.add_argument(
        "--val-log",
        type=Path,
        default=Path("outputs/log_val.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/loss"),
    )
    p.add_argument(
        "--output-file",
        type=Path,
        default=Path("loss_curve_readable.png"),
    )
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--show", action="store_true")
    a = p.parse_args()

    train_path = input_path(
        a.run_dir,
        a.train_log,
        "Training log",
    )
    val_path = input_path(
        a.run_dir,
        a.val_log,
        "Validation log",
    )

    train = np.atleast_1d(
        np.genfromtxt(
            train_path,
            delimiter=",",
            names=True,
            encoding="utf-8",
        )
    )
    val = np.atleast_1d(
        np.genfromtxt(
            val_path,
            delimiter=",",
            names=True,
            encoding="utf-8",
        )
    )

    train_epoch = column(train, ["epoch"])
    train_loss = column(train, ["loss"])
    val_epoch = column(val, ["epoch"])
    val_loss = column(val, ["loss"])

    best = int(np.nanargmin(val_loss))

    print(f"Train log: {train_path}")
    print(f"Validation log: {val_path}")
    print(f"Training points: {len(train_epoch)}")
    print(f"Validation points: {len(val_epoch)}")
    print(f"Best validation loss = {val_loss[best]:.6g}")
    print(f"Best epoch = {val_epoch[best]:.3f}")

    first_epoch = int(np.floor(np.nanmin(train_epoch)))
    last_epoch = int(np.floor(np.nanmax(train_epoch)))
    n_train_epochs = last_epoch - first_epoch + 1

    n_val_points = len(val_epoch)
    val_points_per_epoch = n_val_points / n_train_epochs

    use_thin_val_line = val_points_per_epoch >= 2.0

    print(f"Training log points: {len(train_epoch)}")
    print(f"Training epochs: {n_train_epochs}")
    print(f"Validation points: {n_val_points}")
    print(f"Validation points per epoch: {val_points_per_epoch:.2f}")

    if use_thin_val_line:
        print("Validation style: thin line (validation points >= 2 x training epochs)")
    else:
        print(
            "Validation style: line with markers "
            "(validation points < 2 x training epochs)"
        )

    positive_losses = np.concatenate(
        [
            train_loss[np.isfinite(train_loss) & (train_loss > 0)],
            val_loss[np.isfinite(val_loss) & (val_loss > 0)],
        ]
    )

    if positive_losses.size == 0:
        raise ValueError("No positive finite loss values were found.")

    epsilon = max(
        positive_losses.min() * 0.1,
        1e-12,
    )

    train_loss_log = np.maximum(train_loss, epsilon)
    val_loss_log = np.maximum(val_loss, epsilon)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 5),
        gridspec_kw={"width_ratios": [1, 1.35]},
    )
    ax_full, ax_zoom = axes

    ax_full.plot(
        train_epoch,
        train_loss_log,
        color="tab:blue",
        linewidth=1.5,
        alpha=0.4,
        label="Train loss",
    )

    plot_validation_loss(
        ax_full,
        val_epoch,
        val_loss_log,
        use_thin_line=use_thin_val_line,
    )

    ax_full.scatter(
        val_epoch[best],
        val_loss_log[best],
        color="crimson",
        s=35,
        zorder=5,
        label="Best validation loss",
    )

    ax_full.set(
        yscale="log",
        xlabel="Epoch",
        ylabel="Loss",
        title="Full training history",
    )

    ax_zoom.plot(
        train_epoch,
        train_loss,
        color="tab:blue",
        linewidth=1.5,
        alpha=0.5,
        label="Train loss",
    )

    plot_validation_loss(
        ax_zoom,
        val_epoch,
        val_loss,
        use_thin_line=use_thin_val_line,
    )

    ax_zoom.scatter(
        val_epoch[best],
        val_loss[best],
        color="crimson",
        s=40,
        zorder=5,
        label="Best validation loss",
    )

    ax_zoom.set(
        xlabel="Epoch",
        ylabel="Loss",
        title="Zoomed loss",
    )

    zoom_min, zoom_max = np.percentile(
        positive_losses,
        [2, 98],
    )

    if zoom_max > zoom_min:
        ax_zoom.set_ylim(
            max(zoom_min * 0.5, 1e-8),
            zoom_max * 1.2,
        )

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle(
        "Training and validation loss",
        fontsize=16,
    )
    fig.tight_layout()

    save(
        fig,
        output_path(
            a.output_dir,
            a.output_file,
        ),
        a.dpi,
    )

    if a.show:
        plt.show()


if __name__ == "__main__":
    main()
