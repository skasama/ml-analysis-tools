"""Create a reproducible train/validation/test split from an HDF5 file.

Edit the settings below and run:

    python examples/full/make_split.py
"""

import logging
from pathlib import Path

import numpy as np

from analysis_tools.data import (
    as_integer_array,
    as_scalar_array,
    as_vector_array,
)
from analysis_tools.io import load_hdf5
from analysis_tools.splitting import (
    balanced_sample_indices,
    energy_cut,
    hit_count_cut,
    keep_event_cut,
    label_cut,
    momentum_cut,
    sample_indices,
    split_indices,
    towall_cut,
    valid_data_cut,
    wall_cut,
)
from analysis_tools.transforms import direction_from_angles

# ============================================================
# Input and output
# ============================================================

INPUT_HDF5 = Path("path/to/input.hy")

OUTPUT_NPZ = Path("outputs/split.npz")
OUTPUT_LOG = OUTPUT_NPZ.with_suffix(".log")


# ============================================================
# HDF5 dataset keys
# ============================================================

LABELS_KEY = "labels"
HIT_COUNTS_KEY = "hit_counts"

# Set to None when the dataset is unavailable or unused.
KEEP_EVENT_KEY: str | None = "keep_event"

POSITIONS_KEY = "positions"
DIRECTIONS_KEY = "directions"
ENERGIES_KEY = "energies"


# ============================================================
# Basic cuts
# ============================================================

ALLOWED_LABELS = [0, 1, 2]

MINIMUM_HITS: int | None = 200
MAXIMUM_HITS: int | None = 11146

USE_KEEP_EVENT = True


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


# ============================================================
# Sampling
# ============================================================

SAMPLE_SIZE: int | None = 100_000

BALANCED_SAMPLING = True
BALANCED_LABELS = [0, 1, 2]

RANDOM_SEED = 4242


# ============================================================
# Split
# ============================================================

TRAIN_FRACTION = 0.8
VALIDATION_FRACTION = 0.1

# ============================================================
# Coordinate settings
# ============================================================

# "angles" or "vectors"
DIRECTION_SOURCE = "vectors"

DIRECTIONS_KEY = "directions"
ANGLES_KEY = "angles"

# Cylinder axis and zenith axis:
#   0: x-axis
#   1: y-axis
#   2: z-axis
DETECTOR_AXIS = 2
ZENITH_AXIS = 2


def create_logger() -> logging.Logger:
    """Create a logger writing to both stdout and a log file."""
    OUTPUT_LOG.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = logging.getLogger("make_split")
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
    """Return the HDF5 datasets needed by the enabled cuts."""
    keys = [
        LABELS_KEY,
        HIT_COUNTS_KEY,
    ]

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


def apply_cut(
    current_mask: np.ndarray,
    local_mask: np.ndarray,
    active_indices: np.ndarray,
    *,
    name: str,
    logger: logging.Logger,
) -> np.ndarray:
    """Apply a cut evaluated only on currently active events."""
    if len(local_mask) != len(active_indices):
        raise ValueError(f"{name} mask length does not match the active event count.")

    updated_mask = current_mask.copy()
    updated_mask[active_indices] &= local_mask

    logger.info(
        "  %-30s %10d",
        name,
        np.count_nonzero(updated_mask),
    )

    return updated_mask


def log_settings(
    logger: logging.Logger,
) -> None:
    """Write all analysis settings to the log."""
    test_fraction = 1.0 - TRAIN_FRACTION - VALIDATION_FRACTION

    logger.info("Input and output:")
    logger.info("  Input HDF5: %s", INPUT_HDF5)
    logger.info("  Output NPZ: %s", OUTPUT_NPZ)
    logger.info("  Output log: %s", OUTPUT_LOG)
    logger.info("  Random seed: %d", RANDOM_SEED)

    logger.info("")
    logger.info("Dataset keys:")
    logger.info("  Labels: %s", LABELS_KEY)
    logger.info("  Hit counts: %s", HIT_COUNTS_KEY)
    logger.info("  keep_event: %s", KEEP_EVENT_KEY)
    logger.info("  Positions: %s", POSITIONS_KEY)
    logger.info(
        "  Direction source: %s",
        DIRECTION_SOURCE,
    )

    if DIRECTION_SOURCE == "angles":
        logger.info("  Angles: %s", ANGLES_KEY)
        logger.info(
            "  Zenith axis: %d",
            ZENITH_AXIS,
        )
    else:
        logger.info("  Directions: %s", DIRECTIONS_KEY)

    logger.info(
        "  Detector axis: %d",
        DETECTOR_AXIS,
    )
    logger.info("  Energies: %s", ENERGIES_KEY)

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
    logger.info(
        "  Energy type: %s",
        ENERGY_TYPE,
    )
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
    logger.info("Sampling:")
    logger.info(
        "  Sample size: %s",
        SAMPLE_SIZE,
    )
    logger.info(
        "  Balanced sampling: %s",
        BALANCED_SAMPLING,
    )
    logger.info(
        "  Balanced labels: %s",
        BALANCED_LABELS,
    )

    logger.info("")
    logger.info("Split:")
    logger.info(
        "  Train fraction: %.6f",
        TRAIN_FRACTION,
    )
    logger.info(
        "  Validation fraction: %.6f",
        VALIDATION_FRACTION,
    )
    logger.info(
        "  Test fraction: %.6f",
        test_fraction,
    )


def log_label_counts(
    logger: logging.Logger,
    title: str,
    indices: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Log event counts for each particle label."""
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


def main() -> None:
    logger = create_logger()

    log_settings(logger)
    logger.info("")

    arrays = load_hdf5(
        INPUT_HDF5,
        keys=required_keys(),
    )

    labels = as_integer_array(
        arrays[LABELS_KEY],
        name=LABELS_KEY,
    )
    hit_counts = as_integer_array(
        arrays[HIT_COUNTS_KEY],
        name=HIT_COUNTS_KEY,
    )

    event_count = len(labels)

    if len(hit_counts) != event_count:
        raise ValueError("labels and hit counts have different lengths.")

    selection_mask = np.ones(
        event_count,
        dtype=bool,
    )

    logger.info("Cut flow:")
    logger.info(
        "  %-30s %10d",
        "Initial events",
        event_count,
    )

    # 1. Label cut
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

    # 2. Hit-count cut
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

    # 3. Optional keep_event cut
    if USE_KEEP_EVENT:
        active_indices = np.flatnonzero(selection_mask)

        keep_event = keep_event_cut(arrays[KEEP_EVENT_KEY])

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

    # Load and standardize optional event quantities.
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

    # 4. Valid-data cut
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

    # 5. Energy cut
    if MINIMUM_ENERGY is not None or MAXIMUM_ENERGY is not None:
        if energies is None:
            raise ValueError("Energy cut is enabled, but energies were not loaded.")

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

    # 6. Momentum cut
    if MINIMUM_MOMENTUM is not None or MAXIMUM_MOMENTUM is not None:
        if energies is None:
            raise ValueError("Momentum cut is enabled, but energies were not loaded.")

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

    # Wall cut
    if MINIMUM_WALL_CM is not None or MAXIMUM_WALL_CM is not None:
        if positions is None:
            raise ValueError("Wall cut is enabled, but positions were not loaded.")

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

    # ToWall cut
    if MINIMUM_TOWALL_CM is not None or MAXIMUM_TOWALL_CM is not None:
        if positions is None or directions is None:
            raise ValueError(
                "ToWall cut is enabled, but positions or directions were not loaded."
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

    candidate_indices = np.flatnonzero(selection_mask).astype(np.int64)

    if len(candidate_indices) == 0:
        raise ValueError("No events remain after all cuts.")

    logger.info("")
    log_label_counts(
        logger,
        "Candidate events",
        candidate_indices,
        labels,
    )

    if BALANCED_SAMPLING:
        selected_indices = balanced_sample_indices(
            candidate_indices,
            labels[candidate_indices],
            balanced_labels=BALANCED_LABELS,
            sample_size=SAMPLE_SIZE,
            seed=RANDOM_SEED,
        )
    else:
        selected_indices = sample_indices(
            candidate_indices,
            sample_size=SAMPLE_SIZE,
            seed=RANDOM_SEED,
        )

    logger.info("")
    logger.info("Sampling result:")
    log_label_counts(
        logger,
        "  Selected events",
        selected_indices,
        labels,
    )

    split = split_indices(
        selected_indices,
        train_fraction=TRAIN_FRACTION,
        validation_fraction=VALIDATION_FRACTION,
        seed=RANDOM_SEED,
        shuffle=True,
    )

    logger.info("")
    logger.info("Final split:")
    log_label_counts(
        logger,
        "  Train",
        split.train,
        labels,
    )
    log_label_counts(
        logger,
        "  Validation",
        split.validation,
        labels,
    )
    log_label_counts(
        logger,
        "  Test",
        split.test,
        labels,
    )

    OUTPUT_NPZ.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT_NPZ,
        train_idxs=split.train,
        val_idxs=split.validation,
        test_idxs=split.test,
        selected_idxs=split.selected,
        candidate_idxs=candidate_indices,
        selection_mask=selection_mask,
        seed=np.int64(RANDOM_SEED),
        train_fraction=np.float64(TRAIN_FRACTION),
        validation_fraction=np.float64(VALIDATION_FRACTION),
    )

    logger.info("")
    logger.info("Saved: %s", OUTPUT_NPZ)
    logger.info("Log: %s", OUTPUT_LOG)


if __name__ == "__main__":
    main()
