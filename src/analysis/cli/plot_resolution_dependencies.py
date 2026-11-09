import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.geometry import opening_angle, towall, wall
from analysis.io import assert_aligned, load_outputs
from analysis.paths import output_path
from analysis.plotting import save
from analysis.regression import inverse_minmax, scalar, xyz


def curve(x, y, bins=20):
    e = np.linspace(np.nanmin(x), np.nanmax(x), bins + 1)
    c = (e[:-1] + e[1:]) / 2
    q = np.full(bins, np.nan)
    for i in range(bins):
        m = (x >= e[i]) & ((x <= e[i + 1]) if i == bins - 1 else (x < e[i + 1]))
        q[i] = np.quantile(np.abs(y[m]), 0.68) if m.sum() >= 10 else np.nan
    return c, q


def main():
    p = argparse.ArgumentParser()
    for k in ("vertex", "direction", "energy"):
        p.add_argument(f"--{k}-run", required=True)
        p.add_argument(f"--{k}-predictions", default="outputs/preds.npy")
        p.add_argument(f"--{k}-targets", default="outputs/targets.npy")
        p.add_argument(f"--{k}-indices", default="outputs/indices.npy")
    p.add_argument(
        "--output-dir", type=Path, default=Path("plots/resolution_dependencies")
    )
    p.add_argument(
        "--output-file", type=Path, default=Path("resolution_vs_wall_towall_energy.png")
    )
    p.add_argument("--dpi", type=int, default=200)
    a = p.parse_args()
    items = []
    for k in ("vertex", "direction", "energy"):
        items.append(
            load_outputs(
                getattr(a, f"{k}_run"),
                getattr(a, f"{k}_predictions"),
                getattr(a, f"{k}_targets"),
                getattr(a, f"{k}_indices"),
            )
        )
    v, d, e = items
    assert_aligned(v, d, e)
    vp = inverse_minmax(xyz(v.predictions), [-1690, -1690, -1810], [1690, 1690, 1810])
    vt = inverse_minmax(xyz(v.targets), [-1690, -1690, -1810], [1690, 1690, 1810])
    dp = -1 + 2 * xyz(d.predictions)
    dt = -1 + 2 * xyz(d.targets)
    ep = inverse_minmax(scalar(e.predictions), 0.5, 2000)
    et = inverse_minmax(scalar(e.targets), 0.5, 2000)
    deps = [
        ("Wall [cm]", wall(vt)),
        ("ToWall [cm]", towall(vt, dt)),
        ("True energy [MeV]", et),
    ]
    errs = [
        ("Vertex [cm]", np.linalg.norm(vp - vt, 1)),
        ("Direction [deg]", opening_angle(dt, dp)),
        ("Energy [MeV]", ep - et),
        ("Relative energy [%]", 100 * (ep - et) / et),
    ]
    fig, ax = plt.subplots(3, 4, figsize=(20, 14))
    for i, (xl, x) in enumerate(deps):
        for j, (yl, y) in enumerate(errs):
            c, q = curve(x, y)
            ax[i, j].plot(c, q, "o-")
            ax[i, j].set(xlabel=xl, ylabel=yl)
            ax[i, j].grid(alpha=0.3)
    fig.tight_layout()
    save(fig, output_path(a.output_dir, a.output_file), a.dpi)


if __name__ == "__main__":
    main()
