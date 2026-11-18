from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis_tools.io import load_npy
from analysis_tools.plotting import plot_vertex_errors, save_figure

FIRST = Path("path/to/first_errors.npy")
SECOND = Path("path/to/second_errors.npy")
OUTPUT = Path("outputs/comparison.png")

first = load_npy(FIRST)
second = load_npy(SECOND)

bins = np.linspace(0.0, 500.0, 101)

figure, axis = plot_vertex_errors(
    first,
    bins=bins,
    label="Method 1",
    color="tab:blue",
)

plot_vertex_errors(
    second,
    ax=axis,
    bins=bins,
    label="Method 2",
    color="tab:orange",
)

axis.legend()
axis.grid(alpha=0.3)

save_figure(figure, OUTPUT, close=True)

plt.close(figure)
