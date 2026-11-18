"""Apply additional cuts while preserving an existing data split.

Edit the settings below and run:

    PYTHONPATH=src python examples/full/recut_split.py
"""

import logging
from pathlib import Path

import numpy as np

from analysis_tools.data import (
    as_integer_array,
    as_scalar_array,
    as_vector_array,
)
from analysis_tools.io import load_hdf5, load_npz
from analysis_tools.splitting import (
    SplitIndices,
    energy_cut,
    hit_count_cut,
    keep_event_cut,
    label_cut,
    momentum_cut,
    towall_cut,
    valid_data_cut,
    validate_split,
    wall_cut,
)
from analysis_tools.transforms import direction_from_angles

# ============================================================
# Input and output
# ============================================================

INPUT_HDF5 = Path("path/to/input.hy")
INPUT_SPLIT = Path("path/to/original_split.npz")

OUTPUT_SPLIT = Path("outputs/recut_split.npz")
OUTPUT_LOG = OUTPUT_SPLIT.with_suffix(".log")


# ============================================================
# HDF5 dataset keys
# ============================================================

LABELS_KEY = "labels"
HIT_COUNTS_KEY = "hit_counts"

KEEP_EVENT_KEY: str | None = "keep_event"

POSITIONS_KEY = "positions"
DIRECTIONS_KEY = "directions"
ANGLES_KEY = "angles"
ENERGIES_KEY = "energies"


# ============================================================
# Coordinate settings
# ============================================================

# "vectors" or "angles"
DIRECTION_SOURCE = "vectors"

# z-axis convention
DETECTOR_AXIS = 2
ZENITH_AXIS = 2


# ============================================================
# Basic cuts
# ============================================================

ALLOWED_LABELS: list[int] | None = [0, 1, 2]

MINIMUM_HITS: int | None = None
MAXIMUM_HITS: int | None = None

USE_KEEP_EVENT = False


# ============================================================
# Energy and momentum cuts
# ============================================================

MINIMUM_ENERGY: float | None = None
MAXIMUM_ENERGY: float | None = None

MINIMUM_MOMENTUM: float | None = None
MAXIMUM_MOMENTUM: float | None = None

# "total" or "kinetic"
ENERGY_TYPE = "total"


# ============================================================
# Geometry cuts
# ============================================================

DETECTOR_RADIUS_CM = 1690.0
DETECTOR_HALF_HEIGHT_CM = 1810.0

MINIMUM_WALL_CM: float | None = None
MAXIMUM_WALL_CM: float | None = None

MINIMUM_TOWALL_CM: float | None = None
MAXIMUM_TOWALL_CM: float | None = None


def create_logger() -> logging.Logger:
    """Create a logger writing to stdout and a log file."""
    OUTPUT_LOG.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = logging.getLogger("recut_split")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        OUTPUT_LOG,
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def required_keys() -> list[str]:
    """Return HDF5 keys required by the enabled cuts."""
    keys = [LABELS_KEY]

    if MINIMUM_HITS is not None or MAXIMUM_HITS is not None:
        keys.append(HIT_COUNTS_KEY)

    if USE_KEEP_EVENT:
        if KEEP_EVENT_KEY is None:
            raise ValueError(
                "KEEP_EVENT_KEY must be defined when USE_KEEP_EVENT is True."
            )
        keys.append(KEEP_EVENT_KEY)

    if (
        MINIMUM_ENERGY is not None
        or MAXIMUM_ENERGY is not None
        or MINIMUM_MOMENTUM is not None
        or MAXIMUM_MOMENTUM is not None
    ):
        keys.append(ENERGIES_KEY)

    if (
        MINIMUM_WALL_CM is not None
        or MAXIMUM_WALL_CM is not None
        or MINIMUM_TOWALL_CM is not None
        or MAXIMUM_TOWALL_CM is not None
    ):
        keys.append(POSITIONS_KEY)

    if MINIMUM_TOWALL_CM is not None or MAXIMUM_TOWALL_CM is not None:
        if DIRECTION_SOURCE == "vectors":
            keys.append(DIRECTIONS_KEY)
        elif DIRECTION_SOURCE == "angles":
            keys.append(ANGLES_KEY)
        else:
            raise ValueError("DIRECTION_SOURCE must be 'vectors' or 'angles'.")

    return list(dict.fromkeys(keys))


def load_parent_split() -> SplitIndices:
    """Load the existing split."""
    arrays = load_npz(
        INPUT_SPLIT,
        keys=[
            "train_idxs",
            "val_idxs",
            "test_idxs",
        ],
    )

    split = SplitIndices(
        train=as_integer_array(
            arrays["train_idxs"],
            name="train_idxs",
        ),
        validation=as_integer_array(
            arrays["val_idxs"],
            name="val_idxs",
        ),
        test=as_integer_array(
            arrays["test_idxs"],
            name="test_idxs",
        ),
    )

    validate_split(split)

    return split


def apply_cut(
    current_mask: np.ndarray,
    local_mask: np.ndarray,
    active_indices: np.ndarray,
    *,
    name: str,
    logger: logging.Logger,
) -> np.ndarray:
    """Apply a cut to currently active events."""
    if len(local_mask) != len(active_indices):
        raise ValueError(f"{name} mask length does not match the active event count.")

    updated = current_mask.copy()
    updated[active_indices] &= local_mask

    logger.info(
        "  %-30s %10d",
        name,
        np.count_nonzero(updated),
    )

    return updated


def preserve_split(
    parent: SplitIndices,
    selection_mask: np.ndarray,
) -> SplitIndices:
    """Apply a global selection mask without changing split membership."""
    return SplitIndices(
        train=parent.train[selection_mask[parent.train]],
        validation=parent.validation[selection_mask[parent.validation]],
        test=parent.test[selection_mask[parent.test]],
    )


def log_label_counts(
    logger: logging.Logger,
    title: str,
    indices: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Log total and per-label event counts."""
    logger.info("%s: %d", title, len(indices))

    if len(indices) == 0:
        return

    unique_labels, counts = np.unique(
        labels[indices],
        return_counts=True,
    )

    for label, count in zip(
        unique_labels,
        counts,
        strict=True,
    ):
        logger.info(
            "    label %d: %d",
            int(label),
            int(count),
        )


def log_settings(logger: logging.Logger) -> None:
    """Log input files and cut settings."""
    logger.info("Input and output:")
    logger.info("  Input HDF5: %s", INPUT_HDF5)
    logger.info("  Parent split: %s", INPUT_SPLIT)
    logger.info("  Output split: %s", OUTPUT_SPLIT)
    logger.info("  Output log: %s", OUTPUT_LOG)

    logger.info("")
    logger.info("Selection:")
    logger.info("  Allowed labels: %s", ALLOWED_LABELS)
    logger.info(
        "  Hit count: [%s, %s]",
        MINIMUM_HITS,
        MAXIMUM_HITS,
    )
    logger.info(
        "  Use keep_event: %s",
        USE_KEEP_EVENT,
    )
    logger.info(
        "  Energy: [%s, %s]",
        MINIMUM_ENERGY,
        MAXIMUM_ENERGY,
    )
    logger.info(
        "  Momentum: [%s, %s] MeV/c",
        MINIMUM_MOMENTUM,
        MAXIMUM_MOMENTUM,
    )
    logger.info("  Energy type: %s", ENERGY_TYPE)
    logger.info(
        "  Wall: [%s, %s] cm",
        MINIMUM_WALL_CM,
        MAXIMUM_WALL_CM,
    )
    logger.info(
        "  ToWall: [%s, %s] cm",
        MINIMUM_TOWALL_CM,
        MAXIMUM_TOWALL_CM,
    )

    logger.info("")
    logger.info("Coordinates:")
    logger.info(
        "  Direction source: %s",
        DIRECTION_SOURCE,
    )
    logger.info(
        "  Detector axis: %d",
        DETECTOR_AXIS,
    )
    logger.info(
        "  Zenith axis: %d",
        ZENITH_AXIS,
    )


def main() -> None:
    logger = create_logger()
    log_settings(logger)

    parent = load_parent_split()

    parent_selected = parent.selected

    if len(np.unique(parent_selected)) != len(parent_selected):
        raise ValueError("The parent split contains overlapping indices.")

    arrays = load_hdf5(
        INPUT_HDF5,
        keys=required_keys(),
    )

    labels = as_integer_array(
        arrays[LABELS_KEY],
        name=LABELS_KEY,
    )

    event_count = len(labels)

    if np.any(parent_selected < 0):
        raise ValueError("The parent split contains negative indices.")

    if np.any(parent_selected >= event_count):
        raise ValueError(
            "The parent split contains indices outside the HDF5 event range."
        )

    # Only parent-split events are eligible for the new split.
    selection_mask = np.zeros(
        event_count,
        dtype=bool,
    )
    selection_mask[parent_selected] = True

    logger.info("")
    logger.info("Cut flow:")
    logger.info(
        "  %-30s %10d",
        "Parent selected events",
        len(parent_selected),
    )

    if ALLOWED_LABELS is not None:
        active_indices = np.flatnonzero(selection_mask)
        selection_mask = apply_cut(
            selection_mask,
            label_cut(
                labels[active_indices],
                ALLOWED_LABELS,
            ),
            active_indices,
            name="Label selection",
            logger=logger,
        )
    else:
        logger.info(
            "  %-30s %10s",
            "Label selection",
            "skipped",
        )

    if MINIMUM_HITS is not None or MAXIMUM_HITS is not None:
        hit_counts = as_integer_array(
            arrays[HIT_COUNTS_KEY],
            name=HIT_COUNTS_KEY,
        )

        active_indices = np.flatnonzero(selection_mask)
        selection_mask = apply_cut(
            selection_mask,
            hit_count_cut(
                hit_counts[active_indices],
                minimum=MINIMUM_HITS,
                maximum=MAXIMUM_HITS,
            ),
            active_indices,
            name="Hit-count selection",
            logger=logger,
        )
    else:
        logger.info(
            "  %-30s %10s",
            "Hit-count selection",
            "skipped",
        )

    if USE_KEEP_EVENT:
        keep_event = keep_event_cut(arrays[KEEP_EVENT_KEY])

        active_indices = np.flatnonzero(selection_mask)
        selection_mask = apply_cut(
            selection_mask,
            keep_event[active_indices],
            active_indices,
            name="keep_event selection",
            logger=logger,
        )
    else:
        logger.info(
            "  %-30s %10s",
            "keep_event selection",
            "skipped",
        )

    energies = None
    positions = None
    directions = None

    if ENERGIES_KEY in arrays:
        energies = as_scalar_array(
            arrays[ENERGIES_KEY],
            name=ENERGIES_KEY,
        ).astype(float)

    if POSITIONS_KEY in arrays:
        positions = as_vector_array(
            arrays[POSITIONS_KEY],
            size=3,
            name=POSITIONS_KEY,
        ).astype(float)

    if DIRECTION_SOURCE == "vectors" and DIRECTIONS_KEY in arrays:
        directions = as_vector_array(
            arrays[DIRECTIONS_KEY],
            size=3,
            name=DIRECTIONS_KEY,
        ).astype(float)

    elif DIRECTION_SOURCE == "angles" and ANGLES_KEY in arrays:
        directions = direction_from_angles(
            arrays[ANGLES_KEY],
            axis=ZENITH_AXIS,
        )

    valid_arrays: list[np.ndarray] = []

    if energies is not None:
        valid_arrays.append(energies)

    if positions is not None:
        valid_arrays.append(positions)

    if directions is not None:
        valid_arrays.append(directions)

    if valid_arrays:
        active_indices = np.flatnonzero(selection_mask)
        selection_mask = apply_cut(
            selection_mask,
            valid_data_cut(*(array[active_indices] for array in valid_arrays)),
            active_indices,
            name="Valid-data selection",
            logger=logger,
        )
    else:
        logger.info(
            "  %-30s %10s",
            "Valid-data selection",
            "skipped",
        )

    if MINIMUM_ENERGY is not None or MAXIMUM_ENERGY is not None:
        if energies is None:
            raise ValueError("Energy cut is enabled, but energies are unavailable.")

        active_indices = np.flatnonzero(selection_mask)
        selection_mask = apply_cut(
            selection_mask,
            energy_cut(
                energies[active_indices],
                minimum=MINIMUM_ENERGY,
                maximum=MAXIMUM_ENERGY,
            ),
            active_indices,
            name="Energy selection",
            logger=logger,
        )
    else:
        logger.info(
            "  %-30s %10s",
            "Energy selection",
            "skipped",
        )

    if MINIMUM_MOMENTUM is not None or MAXIMUM_MOMENTUM is not None:
        if energies is None:
            raise ValueError("Momentum cut is enabled, but energies are unavailable.")

        active_indices = np.flatnonzero(selection_mask)
        selection_mask = apply_cut(
            selection_mask,
            momentum_cut(
                energies[active_indices],
                labels[active_indices],
                energy_type=ENERGY_TYPE,
                minimum=MINIMUM_MOMENTUM,
                maximum=MAXIMUM_MOMENTUM,
            ),
            active_indices,
            name="Momentum selection",
            logger=logger,
        )
    else:
        logger.info(
            "  %-30s %10s",
            "Momentum selection",
            "skipped",
        )

    if MINIMUM_WALL_CM is not None or MAXIMUM_WALL_CM is not None:
        if positions is None:
            raise ValueError("Wall cut is enabled, but positions are unavailable.")

        active_indices = np.flatnonzero(selection_mask)
        selection_mask = apply_cut(
            selection_mask,
            wall_cut(
                positions[active_indices],
                detector_radius=DETECTOR_RADIUS_CM,
                detector_half_height=DETECTOR_HALF_HEIGHT_CM,
                axis=DETECTOR_AXIS,
                minimum=MINIMUM_WALL_CM,
                maximum=MAXIMUM_WALL_CM,
            ),
            active_indices,
            name="Wall selection",
            logger=logger,
        )
    else:
        logger.info(
            "  %-30s %10s",
            "Wall selection",
            "skipped",
        )

    if MINIMUM_TOWALL_CM is not None or MAXIMUM_TOWALL_CM is not None:
        if positions is None or directions is None:
            raise ValueError(
                "ToWall cut is enabled, but positions or directions are unavailable."
            )

        active_indices = np.flatnonzero(selection_mask)
        selection_mask = apply_cut(
            selection_mask,
            towall_cut(
                positions[active_indices],
                directions[active_indices],
                detector_radius=DETECTOR_RADIUS_CM,
                detector_half_height=DETECTOR_HALF_HEIGHT_CM,
                axis=DETECTOR_AXIS,
                minimum=MINIMUM_TOWALL_CM,
                maximum=MAXIMUM_TOWALL_CM,
            ),
            active_indices,
            name="ToWall selection",
            logger=logger,
        )
    else:
        logger.info(
            "  %-30s %10s",
            "ToWall selection",
            "skipped",
        )

    recut = preserve_split(
        parent,
        selection_mask,
    )

    validate_split(
        recut,
        expected_indices=recut.selected,
    )

    logger.info("")
    logger.info("Parent split:")
    log_label_counts(
        logger,
        "  Train",
        parent.train,
        labels,
    )
    log_label_counts(
        logger,
        "  Validation",
        parent.validation,
        labels,
    )
    log_label_counts(
        logger,
        "  Test",
        parent.test,
        labels,
    )

    logger.info("")
    logger.info("Recut split:")
    log_label_counts(
        logger,
        "  Train",
        recut.train,
        labels,
    )
    log_label_counts(
        logger,
        "  Validation",
        recut.validation,
        labels,
    )
    log_label_counts(
        logger,
        "  Test",
        recut.test,
        labels,
    )

    OUTPUT_SPLIT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT_SPLIT,
        train_idxs=recut.train,
        val_idxs=recut.validation,
        test_idxs=recut.test,
        selected_idxs=recut.selected,
        parent_selected_idxs=parent_selected,
        selection_mask=selection_mask,
    )

    logger.info("")
    logger.info("Saved: %s", OUTPUT_SPLIT)
    logger.info("Log: %s", OUTPUT_LOG)


if __name__ == "__main__":
    main()
