import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from analysis.paths import output_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5file", type=Path, required=True)
    p.add_argument("--geometry-file", type=Path, required=True)
    p.add_argument("--event-index-dataset", default="event_hits_index")
    p.add_argument("--pmt-dataset", default="hit_pmt")
    p.add_argument("--time-dataset", default="hit_time")
    p.add_argument("--charge-dataset", default="hit_charge")
    p.add_argument("--position-key", default="position")
    p.add_argument("--event", type=int, required=True)
    p.add_argument("--channel", choices=["charge", "time"], default="charge")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dpi", type=int, default=150)
    a = p.parse_args()
    with np.load(a.geometry_file) as g:
        pos = np.asarray(g[a.position_key], float)
    with h5py.File(a.h5file) as h:
        idx = np.asarray(h[a.event_index_dataset])
        start, stop = int(idx[a.event]), int(idx[a.event + 1])
        pmt = np.asarray(h[a.pmt_dataset][start:stop], int)
        value = np.asarray(
            h[a.charge_dataset if a.channel == "charge" else a.time_dataset][
                start:stop
            ],
            float,
        )
    hit = pos[pmt]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(hit[:, 0], hit[:, 1], hit[:, 2], c=value, s=10, cmap="plasma")
    fig.colorbar(sc, ax=ax, label=a.channel)
    ax.set(xlabel="x", ylabel="y", zlabel="z", title=f"Event {a.event}")
    path = output_path(".", a.output)
    fig.savefig(path, dpi=a.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
