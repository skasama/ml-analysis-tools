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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--train-log", type=Path, default=Path("outputs/log_train_0.csv"))
    p.add_argument("--val-log", type=Path, default=Path("outputs/log_val.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("plots/loss"))
    p.add_argument("--output-file", type=Path, default=Path("loss_curve_readable.png"))
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--show", action="store_true")
    a = p.parse_args()
    train_path = input_path(a.run_dir, a.train_log, "Training log")
    val_path = input_path(a.run_dir, a.val_log, "Validation log")
    train = np.atleast_1d(
        np.genfromtxt(train_path, delimiter=",", names=True, encoding="utf-8")
    )
    val = np.atleast_1d(
        np.genfromtxt(val_path, delimiter=",", names=True, encoding="utf-8")
    )
    te, tl = column(train, ["epoch"]), column(train, ["loss"])
    ve, vl = column(val, ["epoch"]), column(val, ["loss"])
    best = int(np.nanargmin(vl))
    print(f"Train log: {train_path}")
    print(f"Validation log: {val_path}")
    print(f"Best validation loss = {vl[best]:.6g}")
    print(f"Best epoch = {ve[best]:.3f}")
    pos = np.concatenate([tl[tl > 0], vl[vl > 0]])
    eps = max(pos.min() * 0.1, 1e-12)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(te, np.maximum(tl, eps), label="Train loss", alpha=0.4)
    ax[0].plot(ve, np.maximum(vl, eps), "o-", label="Validation loss")
    ax[0].scatter(ve[best], max(vl[best], eps), c="crimson", label="Best")
    ax[0].set(
        yscale="log", xlabel="Epoch", ylabel="Loss", title="Full training history"
    )
    ax[1].plot(te, tl, label="Train loss", alpha=0.5)
    ax[1].plot(ve, vl, "o-", label="Validation loss")
    ax[1].scatter(ve[best], vl[best], c="crimson")
    ax[1].set(xlabel="Epoch", ylabel="Loss", title="Zoomed loss")
    lo, hi = np.percentile(pos, [2, 98])
    ax[1].set_ylim(max(lo * 0.5, 1e-8), hi * 1.2)
    for x in ax:
        x.grid(alpha=0.3)
        x.legend()
    fig.tight_layout()
    save(fig, output_path(a.output_dir, a.output_file), a.dpi)
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()
