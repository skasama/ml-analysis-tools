"""Apply additional cuts while preserving an existing data split.

Particle labels:

    0: muon
    1: electron
    2: charged pion

Edit the settings below and run:

    uv run python examples/full/recut_split.py
"""

import logging
from pathlib import Path

import h5py
import numpy as np

from analysis_tools.data import (
    as_integer_array,
    as_scalar_array,
    as_vector_array,
    hit_counts_from_indices,
)
from analysis_tools.io import (
    load_hdf5,
    load_npz,
)
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
from analysis_tools.transforms import (
    direction_from_angles,
)

# Input and output
INPUT_HDF5 = Path("path/to/input.hy")
INPUT_SPLIT = Path("path/to/original_split.npz")

OUTPUT_SPLIT = Path("outputs/recut_split.npz")
OUTPUT_LOG = OUTPUT_SPLIT.with_suffix(".log")

# HDF5 dataset keys
LABELS_KEY = "labels"

EVENT_HITS_INDEX_KEY = "event_hits_index"
HIT_PMT_KEY = "hit_pmt"

KEEP_EVENT_KEY: str | None = "keep_event"

POSITIONS_KEY = "positions"
DIRECTIONS_KEY = "directions"
ANGLES_KEY = "angles"
ENERGIES_KEY = "energies"

# Particle labels
# 0: muon
# 1: electron
# 2: charged pion
ALLOWED_LABELS: list[int] | None = [
    0,
    1,
    2,
]

# Basic cuts
MINIMUM_HITS: int | None = None
MAXIMUM_HITS: int | None = None

USE_KEEP_EVENT = False

# Energy and momentum cuts
MINIMUM_ENERGY: float | None = None
MAXIMUM_ENERGY: float | None = None

MINIMUM_MOMENTUM: float | None = None
MAXIMUM_MOMENTUM: float | None = None

# "total" or "kinetic"
ENERGY_TYPE = "total"

# Geometry cuts
DETECTOR_RADIUS_CM = 1690.0
DETECTOR_HALF_HEIGHT_CM = 1810.0

MINIMUM_WALL_CM: float | None = None
MAXIMUM_WALL_CM: float | None = None

MINIMUM_TOWALL_CM: float | None = None
MAXIMUM_TOWALL_CM: float | None = None

# Coordinate settings
# Use:
#
#   "directions": read the HDF5 "directions" dataset
#   "angles": read the HDF5 "angles" dataset and convert it
DIRECTION_SOURCE = "directions"

# z-axis
DETECTOR_AXIS = 2
ZENITH_AXIS = 2


def hit_cut_enabled() -> bool:
    """Return whether the hit-count cut is enabled."""
    return MINIMUM_HITS is not None or MAXIMUM_HITS is not None


def energy_cut_enabled() -> bool:
    """Return whether the energy cut is enabled."""
    return MINIMUM_ENERGY is not None or MAXIMUM_ENERGY is not None


def momentum_cut_enabled() -> bool:
    """Return whether the momentum cut is enabled."""
    return MINIMUM_MOMENTUM is not None or MAXIMUM_MOMENTUM is not None


def wall_cut_enabled() -> bool:
    """Return whether the Wall cut is enabled."""
    return MINIMUM_WALL_CM is not None or MAXIMUM_WALL_CM is not None


def towall_cut_enabled() -> bool:
    """Return whether the ToWall cut is enabled."""
    return MINIMUM_TOWALL_CM is not None or MAXIMUM_TOWALL_CM is not None


def create_logger() -> logging.Logger:
    """Create a logger writing to the terminal and a log file."""
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


def validate_settings() -> None:
    """Validate settings before opening input files."""
    if USE_KEEP_EVENT and KEEP_EVENT_KEY is None:
        raise ValueError("KEEP_EVENT_KEY must be defined when USE_KEEP_EVENT is True.")

    if DIRECTION_SOURCE not in {
        "directions",
        "angles",
    }:
        raise ValueError("DIRECTION_SOURCE must be 'directions' or 'angles'.")

    if ENERGY_TYPE not in {
        "total",
        "kinetic",
    }:
        raise ValueError("ENERGY_TYPE must be 'total' or 'kinetic'.")


def required_keys() -> list[str]:
    """Return HDF5 datasets required by enabled cuts."""
    keys = [
        LABELS_KEY,
    ]

    if hit_cut_enabled():
        keys.append(EVENT_HITS_INDEX_KEY)

    if USE_KEEP_EVENT:
        if KEEP_EVENT_KEY is None:
            raise ValueError(
                "KEEP_EVENT_KEY must be defined when USE_KEEP_EVENT is True."
            )

        keys.append(KEEP_EVENT_KEY)

    if energy_cut_enabled() or momentum_cut_enabled():
        keys.append(ENERGIES_KEY)

    if wall_cut_enabled() or towall_cut_enabled():
        keys.append(POSITIONS_KEY)

    if towall_cut_enabled():
        if DIRECTION_SOURCE == "directions":
            keys.append(DIRECTIONS_KEY)
        else:
            keys.append(ANGLES_KEY)

    return list(dict.fromkeys(keys))


def load_total_hit_count() -> int:
    """Read the total number of hits without loading hit_pmt."""
    with h5py.File(
        INPUT_HDF5,
        "r",
    ) as h5_file:
        if HIT_PMT_KEY not in h5_file:
            raise KeyError(f"Missing dataset '{HIT_PMT_KEY}' in {INPUT_HDF5}.")

        return int(h5_file[HIT_PMT_KEY].shape[0])


def load_parent_split() -> SplitIndices:
    """Load and validate the existing split."""
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
    """Apply a cut evaluated on currently active events."""
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


def log_skipped_cut(
    logger: logging.Logger,
    name: str,
) -> None:
    """Log that a cut was skipped."""
    logger.info(
        "  %-30s %10s",
        name,
        "skipped",
    )


def preserve_split(
    parent: SplitIndices,
    selection_mask: np.ndarray,
) -> SplitIndices:
    """Apply a mask without changing split membership."""
    return SplitIndices(
        train=parent.train[selection_mask[parent.train]],
        validation=parent.validation[selection_mask[parent.validation]],
        test=parent.test[selection_mask[parent.test]],
    )


def validate_recut_membership(
    parent: SplitIndices,
    recut: SplitIndices,
) -> None:
    """Verify that recutting preserved split membership."""
    validate_split(recut)

    train_is_subset = np.all(
        np.isin(
            recut.train,
            parent.train,
        )
    )

    if not train_is_subset:
        raise ValueError(
            "Recut train indices are not a subset of the parent train split."
        )

    validation_is_subset = np.all(
        np.isin(
            recut.validation,
            parent.validation,
        )
    )

    if not validation_is_subset:
        raise ValueError(
            "Recut validation indices are not a subset of the parent validation split."
        )

    test_is_subset = np.all(
        np.isin(
            recut.test,
            parent.test,
        )
    )

    if not test_is_subset:
        raise ValueError(
            "Recut test indices are not a subset of the parent test split."
        )


def log_label_counts(
    logger: logging.Logger,
    title: str,
    indices: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Log total and per-label event counts."""
    logger.info(
        "%s: %d",
        title,
        len(indices),
    )

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


def log_settings(
    logger: logging.Logger,
) -> None:
    """Write all settings to the log."""
    logger.info("Input and output:")
    logger.info(
        "  Input HDF5: %s",
        INPUT_HDF5,
    )
    logger.info(
        "  Parent split: %s",
        INPUT_SPLIT,
    )
    logger.info(
        "  Output split: %s",
        OUTPUT_SPLIT,
    )
    logger.info(
        "  Output log: %s",
        OUTPUT_LOG,
    )

    logger.info("")
    logger.info("Particle labels:")
    logger.info("  0: muon")
    logger.info("  1: electron")
    logger.info("  2: charged pion")

    logger.info("")
    logger.info("Dataset keys:")
    logger.info(
        "  Labels: %s",
        LABELS_KEY,
    )

    if hit_cut_enabled():
        logger.info(
            "  Event hit indices: %s",
            EVENT_HITS_INDEX_KEY,
        )
        logger.info(
            "  Hit PMTs: %s (shape only)",
            HIT_PMT_KEY,
        )
        logger.info("  Hit counts: calculated from event boundaries")
    else:
        logger.info("  Hit-count datasets: not required")

    logger.info(
        "  keep_event: %s",
        KEEP_EVENT_KEY,
    )
    logger.info(
        "  Positions: %s",
        POSITIONS_KEY,
    )
    logger.info(
        "  Energies: %s",
        ENERGIES_KEY,
    )
    logger.info(
        "  Direction source: %s",
        DIRECTION_SOURCE,
    )

    if DIRECTION_SOURCE == "directions":
        logger.info(
            "  Directions: %s",
            DIRECTIONS_KEY,
        )
    else:
        logger.info(
            "  Angles: %s",
            ANGLES_KEY,
        )
        logger.info(
            "  Zenith axis: %d",
            ZENITH_AXIS,
        )

    logger.info(
        "  Detector axis: %d",
        DETECTOR_AXIS,
    )

    logger.info("")
    logger.info("Selection:")
    logger.info(
        "  Allowed labels: %s",
        ALLOWED_LABELS,
    )
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
        "  Energy: [%s, %s] MeV",
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


def main() -> None:
    """Apply cuts and save a recut split."""
    validate_settings()

    logger = create_logger()
    log_settings(logger)

    # ========================================================
    # Load and validate the parent split
    # ========================================================

    parent = load_parent_split()
    parent_selected = parent.selected

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

    # Only events in the parent split are eligible.
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

    # ========================================================
    # 1. Label cut
    # ========================================================

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
        log_skipped_cut(
            logger,
            "Label selection",
        )

    # ========================================================
    # 2. Hit-count cut
    # ========================================================

    if hit_cut_enabled():
        event_hits_index = as_integer_array(
            arrays[EVENT_HITS_INDEX_KEY],
            name=EVENT_HITS_INDEX_KEY,
        )

        total_hit_count = load_total_hit_count()

        hit_counts = hit_counts_from_indices(
            event_hits_index,
            total_hit_count,
        )

        if len(hit_counts) != event_count:
            raise ValueError(
                "labels and calculated hit counts "
                "have different lengths: "
                f"labels={event_count}, "
                f"hit_counts={len(hit_counts)}."
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
        log_skipped_cut(
            logger,
            "Hit-count selection",
        )

    # ========================================================
    # 3. Optional keep_event cut
    # ========================================================

    if USE_KEEP_EVENT:
        if KEEP_EVENT_KEY is None:
            raise ValueError("KEEP_EVENT_KEY must be defined.")

        keep_event = keep_event_cut(arrays[KEEP_EVENT_KEY])

        if len(keep_event) != event_count:
            raise ValueError("labels and keep_event have different lengths.")

        active_indices = np.flatnonzero(selection_mask)

        selection_mask = apply_cut(
            selection_mask,
            keep_event[active_indices],
            active_indices,
            name="keep_event selection",
            logger=logger,
        )

    else:
        log_skipped_cut(
            logger,
            "keep_event selection",
        )

    # ========================================================
    # Load optional event quantities
    # ========================================================

    energies = None
    positions = None
    directions = None

    if ENERGIES_KEY in arrays:
        energies = as_scalar_array(
            arrays[ENERGIES_KEY],
            name=ENERGIES_KEY,
        ).astype(float)

        if len(energies) != event_count:
            raise ValueError("labels and energies have different lengths.")

    if POSITIONS_KEY in arrays:
        positions = as_vector_array(
            arrays[POSITIONS_KEY],
            size=3,
            name=POSITIONS_KEY,
        ).astype(float)

        if len(positions) != event_count:
            raise ValueError("labels and positions have different lengths.")

    if DIRECTION_SOURCE == "directions" and DIRECTIONS_KEY in arrays:
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

    if directions is not None and len(directions) != event_count:
        raise ValueError("labels and directions have different lengths.")

    # ========================================================
    # 4. Valid-data cut
    # ========================================================

    valid_arrays: list[np.ndarray] = []

    if energies is not None:
        valid_arrays.append(energies)

    if positions is not None:
        valid_arrays.append(positions)

    if directions is not None:
        valid_arrays.append(directions)

    if valid_arrays:
        active_indices = np.flatnonzero(selection_mask)

        active_arrays = [array[active_indices] for array in valid_arrays]

        selection_mask = apply_cut(
            selection_mask,
            valid_data_cut(*active_arrays),
            active_indices,
            name="Valid-data selection",
            logger=logger,
        )

    else:
        log_skipped_cut(
            logger,
            "Valid-data selection",
        )

    # ========================================================
    # 5. Energy cut
    # ========================================================

    if energy_cut_enabled():
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
        log_skipped_cut(
            logger,
            "Energy selection",
        )

    # ========================================================
    # 6. Momentum cut
    # ========================================================

    if momentum_cut_enabled():
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
        log_skipped_cut(
            logger,
            "Momentum selection",
        )

    # ========================================================
    # 7. Wall cut
    # ========================================================

    if wall_cut_enabled():
        if positions is None:
            raise ValueError("Wall cut is enabled, but positions were not loaded.")

        active_indices = np.flatnonzero(selection_mask)

        selection_mask = apply_cut(
            selection_mask,
            wall_cut(
                positions[active_indices],
                detector_radius=(DETECTOR_RADIUS_CM),
                detector_half_height=(DETECTOR_HALF_HEIGHT_CM),
                axis=DETECTOR_AXIS,
                minimum=MINIMUM_WALL_CM,
                maximum=MAXIMUM_WALL_CM,
            ),
            active_indices,
            name="Wall selection",
            logger=logger,
        )

    else:
        log_skipped_cut(
            logger,
            "Wall selection",
        )

    # ========================================================
    # 8. ToWall cut
    # ========================================================

    if towall_cut_enabled():
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
                detector_radius=(DETECTOR_RADIUS_CM),
                detector_half_height=(DETECTOR_HALF_HEIGHT_CM),
                axis=DETECTOR_AXIS,
                minimum=MINIMUM_TOWALL_CM,
                maximum=MAXIMUM_TOWALL_CM,
            ),
            active_indices,
            name="ToWall selection",
            logger=logger,
        )

    else:
        log_skipped_cut(
            logger,
            "ToWall selection",
        )

    # ========================================================
    # Preserve the parent split membership
    # ========================================================

    recut = preserve_split(
        parent,
        selection_mask,
    )

    validate_recut_membership(
        parent,
        recut,
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

    # ========================================================
    # Save
    # ========================================================

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
        parent_train_idxs=parent.train,
        parent_val_idxs=parent.validation,
        parent_test_idxs=parent.test,
        parent_selected_idxs=(parent_selected),
        selection_mask=selection_mask,
    )

    logger.info("")
    logger.info(
        "Saved: %s",
        OUTPUT_SPLIT,
    )
    logger.info(
        "Log: %s",
        OUTPUT_LOG,
    )


if __name__ == "__main__":
    main()
