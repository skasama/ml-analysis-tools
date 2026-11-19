"""Load and inspect common analysis data formats.

Enable the sections you want to run, edit the paths below, and run:

    uv run python examples/simple/load_data.py

In Apptainer:

    source /home/skasama/ml-analysis-tools/setup_apptainer_env.sh
    python examples/simple/load_data.py
"""

from pathlib import Path

import numpy as np

from analysis_tools.io import (
    list_hdf5_datasets,
    load_array_pair,
    load_hdf5,
    load_npy,
    load_npz,
)

# ============================================================
# Select examples to run
# ============================================================

LOAD_PREDICTION_TARGET_PAIR = True
LOAD_SINGLE_NPY = False
LOAD_NPZ = False
INSPECT_HDF5 = False
LOAD_HDF5_DATASETS = False


# ============================================================
# Input paths
# ============================================================

PREDICTIONS_PATH = Path("path/to/outputs/preds.npy")
TARGETS_PATH = Path("path/to/outputs/targets.npy")

NPY_PATH = Path("path/to/array.npy")
NPZ_PATH = Path("path/to/data.npz")
HDF5_PATH = Path("path/to/data.hy")


# ============================================================
# NPZ and HDF5 keys
# ============================================================

NPZ_KEYS = [
    "train_idxs",
    "val_idxs",
    "test_idxs",
]

HDF5_KEYS = [
    "labels",
    "positions",
    "directions",
    "energies",
]


def print_array_summary(
    name: str,
    array: np.ndarray,
) -> None:
    """Print basic array information."""
    print(f"\n{name}")
    print(f"  Shape: {array.shape}")
    print(f"  Dtype: {array.dtype}")
    print(f"  Events: {len(array):,}")

    if np.issubdtype(array.dtype, np.number):
        finite = np.isfinite(array)

        print(f"  Finite values: {np.count_nonzero(finite):,} / {array.size:,}")

        if np.any(finite):
            finite_values = array[finite]

            print(
                f"  Range: {np.min(finite_values):.6g} to {np.max(finite_values):.6g}"
            )

    preview_size = min(3, len(array))

    print("  First entries:")
    print(array[:preview_size])


def load_prediction_target_example() -> None:
    """Load prediction and target NumPy arrays."""
    predictions, targets = load_array_pair(
        PREDICTIONS_PATH,
        TARGETS_PATH,
        first_name="predictions",
        second_name="targets",
    )

    print_array_summary(
        "Predictions",
        predictions,
    )
    print_array_summary(
        "Targets",
        targets,
    )


def load_single_npy_example() -> None:
    """Load one NumPy array."""
    array = load_npy(NPY_PATH)

    print_array_summary(
        NPY_PATH.name,
        array,
    )


def load_npz_example() -> None:
    """Load selected arrays from an NPZ file."""
    arrays = load_npz(
        NPZ_PATH,
        keys=NPZ_KEYS,
    )

    for key, array in arrays.items():
        print_array_summary(
            key,
            array,
        )


def inspect_hdf5_example() -> None:
    """List all datasets stored in an HDF5 file."""
    datasets = list_hdf5_datasets(HDF5_PATH)

    print(f"\nHDF5 file: {HDF5_PATH}")
    print(f"Datasets: {len(datasets)}")

    for key in datasets:
        print(f"  {key}")


def load_hdf5_example() -> None:
    """Load selected datasets from an HDF5 file."""
    arrays = load_hdf5(
        HDF5_PATH,
        keys=HDF5_KEYS,
    )

    for key, array in arrays.items():
        print_array_summary(
            key,
            array,
        )


def main() -> None:
    if LOAD_PREDICTION_TARGET_PAIR:
        load_prediction_target_example()

    if LOAD_SINGLE_NPY:
        load_single_npy_example()

    if LOAD_NPZ:
        load_npz_example()

    if INSPECT_HDF5:
        inspect_hdf5_example()

    if LOAD_HDF5_DATASETS:
        load_hdf5_example()


if __name__ == "__main__":
    main()
