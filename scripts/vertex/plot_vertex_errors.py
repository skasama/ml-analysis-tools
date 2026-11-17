"""Plot three-dimensional vertex errors from prediction and target arrays."""

from pathlib import Path

import numpy as np

from analysis_tools.data import as_vector_array
from analysis_tools.io import load_array_pair
from analysis_tools.metrics import vertex_errors
from analysis_tools.plotting import plot_vertex_errors, save_figure
from analysis_tools.transforms import MinMaxTransform

# Input
PREDICTIONS_PATH = Path("path/to/preds.npy")
TARGETS_PATH = Path("path/to/targets.npy")

# Output
OUTPUT_PATH = Path("outputs/vertex_errors.png")

# Vertex normalization range [cm]
VERTEX_MINIMUM = [-1690.0, -1690.0, -1810.0]
VERTEX_MAXIMUM = [1690.0, 1690.0, 1810.0]

# Plot settings
BINS = 100
X_LIMITS = (0.0, 500.0)


def main() -> None:
    predictions, targets = load_array_pair(
        PREDICTIONS_PATH,
        TARGETS_PATH,
        first_name="predictions",
        second_name="targets",
    )

    predictions = as_vector_array(
        predictions,
        size=3,
        name="predictions",
    )
    targets = as_vector_array(
        targets,
        size=3,
        name="targets",
    )

    transform = MinMaxTransform(
        minimum=VERTEX_MINIMUM,
        maximum=VERTEX_MAXIMUM,
    )

    predictions_cm = transform.inverse(predictions)
    targets_cm = transform.inverse(targets)

    errors_cm = vertex_errors(
        predictions_cm,
        targets_cm,
    )

    figure, axis = plot_vertex_errors(
        errors_cm,
        bins=BINS,
        xlabel="3D vertex error [cm]",
        xlim=X_LIMITS,
        label="ML",
    )

    axis.legend()

    save_figure(
        figure,
        OUTPUT_PATH,
        close=True,
    )

    print(f"Events: {len(errors_cm):,}")
    print(f"68% quantile: {np.quantile(errors_cm, 0.68):.2f} cm")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
