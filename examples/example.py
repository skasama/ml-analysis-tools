"""Example: compare ML and non-ML resolution dependencies.

Edit the paths and variable names below, then run:

    python examples/example.py

The ML result uses the standard outputs/{preds,targets}.npy layout.
The ``others`` result reads prediction and truth arrays from one HDF5-compatible
``.hy`` file. Wall and ToWall are calculated from the truth arrays; no derived
wall/towall files are required.
"""

import sys
from pathlib import Path

from analysis.cli.compare_resolution_dependencies import main

# ---------------------------------------------------------------------------
# Edit this section for your analysis.
# ---------------------------------------------------------------------------

ML_VERTEX_RUN = Path("/path/to/ml_vertex_run")
ML_DIRECTION_RUN = Path("/path/to/ml_direction_run")
ML_ENERGY_RUN = Path("/path/to/ml_energy_run")

OTHERS_RUN = Path("/path/to/others")
OTHERS_FILE = Path("result.hy")

OTHERS_VERTEX_PREDICTION = "fitqun/e_1rpos"
OTHERS_VERTEX_TRUTH = "/multi/positions"
OTHERS_DIRECTION_PREDICTION = "fitqun/e_1rdir"
OTHERS_DIRECTION_TRUTH = "multi/directions"
OTHERS_ENERGY_PREDICTION = "fitqun/e_1rmom"
OTHERS_ENERGY_TRUTH = "multi/energies"

OUTPUT_DIR = Path("plots/resolution_dependencies_comparison")


def build_arguments():
    """Build the command-line arguments for the comparison CLI."""
    return [
        "analysis",
        "compare-resolution-dependencies",
        "--label",
        "cls-GAT",
        "--vertex-run",
        str(ML_VERTEX_RUN),
        "--vertex-predictions",
        "outputs/preds.npy",
        "--vertex-targets",
        "outputs/targets.npy",
        "--direction-run",
        str(ML_DIRECTION_RUN),
        "--direction-predictions",
        "outputs/preds.npy",
        "--direction-targets",
        "outputs/targets.npy",
        "--energy-run",
        str(ML_ENERGY_RUN),
        "--energy-predictions",
        "outputs/preds.npy",
        "--energy-targets",
        "outputs/targets.npy",
        "--label",
        "others",
        "--already-physical",
        "others",
        "--vertex-run",
        str(OTHERS_RUN),
        "--vertex-predictions",
        str(OTHERS_FILE),
        "--vertex-predictions-variable",
        OTHERS_VERTEX_PREDICTION,
        "--vertex-targets",
        str(OTHERS_FILE),
        "--vertex-targets-variable",
        OTHERS_VERTEX_TRUTH,
        "--direction-run",
        str(OTHERS_RUN),
        "--direction-predictions",
        str(OTHERS_FILE),
        "--direction-predictions-variable",
        OTHERS_DIRECTION_PREDICTION,
        "--direction-targets",
        str(OTHERS_FILE),
        "--direction-targets-variable",
        OTHERS_DIRECTION_TRUTH,
        "--energy-run",
        str(OTHERS_RUN),
        "--energy-predictions",
        str(OTHERS_FILE),
        "--energy-predictions-variable",
        OTHERS_ENERGY_PREDICTION,
        "--energy-targets",
        str(OTHERS_FILE),
        "--energy-targets-variable",
        OTHERS_ENERGY_TRUTH,
        "--output-dir",
        str(OUTPUT_DIR),
    ]


if __name__ == "__main__":
    sys.argv = build_arguments()
    main()
