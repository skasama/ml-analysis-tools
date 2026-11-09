import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.io import load_outputs
from analysis.paths import output_path
from analysis.plotting import save
from analysis.regression import inverse_minmax, xyz


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--predictions", default="outputs/preds.npy")
    p.add_argument("--targets", default="outputs/targets.npy")
    p.add_argument("--indices", default="outputs/indices.npy")
    p.add_argument("--output-dir", type=Path, default=Path("plots/vertex"))
    p.add_argument("--output-file", type=Path, default=None)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--minimum", nargs=3, type=float, default=[-1690, -1690, -1810])
    p.add_argument("--maximum", nargs=3, type=float, default=[1690, 1690, 1810])
    p.add_argument("--already-cm", action="store_true")
    a = p.parse_args()
    o = load_outputs(a.run_dir, a.predictions, a.targets, a.indices)
    pred = (
        inverse_minmax(xyz(o.predictions), a.minimum, a.maximum)
        if not a.already_cm
        else xyz(o.predictions)
    )
    true = (
        inverse_minmax(xyz(o.targets), a.minimum, a.maximum)
        if not a.already_cm
        else xyz(o.targets)
    )
    error = np.linalg.norm(pred - true, axis=1)
    xlabel = "3D position error [cm]"
    name = "position_error_3d.png"
    q = np.quantile(error, 0.68)
    print(f"Events: {len(error):,}; 68% quantile = {q:.4f}")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(error, bins=100, histtype="step")
    ax.axvline(q, c="crimson", ls="--", label=f"68% = {q:.3f}")
    ax.set(xlabel=xlabel, ylabel="Events")
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, output_path(a.output_dir, a.output_file or name), a.dpi)


if __name__ == "__main__":
    main()
