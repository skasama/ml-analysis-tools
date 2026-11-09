import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.io import load_outputs
from analysis.paths import output_path
from analysis.plotting import save


def labels(x):
    x = np.asarray(x)
    return (
        x.astype(int)
        if x.ndim == 1
        else (x[:, 0].astype(int) if x.shape[1] == 1 else np.argmax(x, 1))
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--classification-run", required=True)
    p.add_argument("--predictions", default="outputs/preds.npy")
    p.add_argument("--targets", default="outputs/targets.npy")
    p.add_argument("--indices", default="outputs/indices.npy")
    p.add_argument("--truth-file", type=Path, required=True)
    p.add_argument(
        "--output-dir", type=Path, default=Path("plots/classification_dependencies")
    )
    p.add_argument(
        "--output-file",
        type=Path,
        default=Path("classification_mispid_dependencies.png"),
    )
    p.add_argument("--dpi", type=int, default=200)
    a = p.parse_args()
    o = load_outputs(a.classification_run, a.predictions, a.targets, a.indices)
    y = labels(o.targets)
    pred = np.argmax(o.predictions, 1)
    with np.load(a.truth_file, allow_pickle=False) as z:
        vars = [
            ("Wall [cm]", z["wall"]),
            ("ToWall [cm]", z["towall"]),
            ("Energy [MeV]", z["energies"]),
        ]
        truth = labels(z["labels"])
    if not np.array_equal(y, truth):
        raise ValueError("Truth labels and output targets are not aligned")
    fig, ax = plt.subplots(3, 3, figsize=(16, 13))
    names = ["muon", "electron", "pion"]

    for c in range(3):
        for j, (xl, x) in enumerate(vars):
            e = np.linspace(np.nanmin(x), np.nanmax(x), 21)
            cent = (e[:-1] + e[1:]) / 2
            rate = np.full(20, np.nan)
            for i in range(20):
                m = (
                    (y == c)
                    & (x >= e[i])
                    & ((x <= e[i + 1]) if i == 19 else (x < e[i + 1]))
                )
                rate[i] = np.mean(pred[m] != c) if m.sum() >= 10 else np.nan
            ax[c, j].plot(cent, 100 * rate, "o-")
            ax[c, j].set(xlabel=xl, ylabel="Mis-PID rate [%]", title=f"True {names[c]}")
            ax[c, j].grid(alpha=0.3)
    fig.tight_layout()
    save(fig, output_path(a.output_dir, a.output_file), a.dpi)


if __name__ == "__main__":
    main()
