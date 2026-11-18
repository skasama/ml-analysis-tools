from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from analysis_tools.data import (
    as_integer_array,
    as_scalar_array,
    as_vector_array,
    validate_same_length,
)
from analysis_tools.physics import (
    energy_to_momentum,
    kinetic_energy_to_momentum,
)
from analysis_tools.transforms.geometry import (
    compute_towall,
    compute_wall,
)


def valid_data_cut(
    *arrays: ArrayLike,
) -> NDArray[np.bool_]:
    """

    Select events containing finite values in all arrays.

    Parameters
    ----------
    *arrays : ArrayLike
        One or more arrays to check for finite values.

    Returns
    -------
    NDArray[np.bool_]
        Boolean mask indicating events with finite values in all arrays.
    """
    if not arrays:
        raise ValueError("At least one array is required.")

    converted = [np.asarray(array) for array in arrays]
    validate_same_length(*converted)

    mask = np.ones(
        len(converted[0]),
        dtype=bool,
    )

    for array in converted:
        if not np.issubdtype(
            array.dtype,
            np.number,
        ):
            raise ValueError("valid_data_cut accepts only numeric arrays.")

        if array.ndim == 1:
            mask &= np.isfinite(array)
        else:
            event_axes = tuple(range(1, array.ndim))
            mask &= np.all(
                np.isfinite(array),
                axis=event_axes,
            )

    return mask


def label_cut(
    labels: ArrayLike,
    allowed_labels: Sequence[int],
) -> NDArray[np.bool_]:
    """

    Select events with one of the requested particle labels.

    Parameters
    ----------
    labels : ArrayLike
        Array of particle labels for each event.
    allowed_labels : Sequence[int]
        Sequence of allowed particle labels.

    Returns
    -------
    NDArray[np.bool_]
        Boolean mask indicating events with allowed particle labels.
    """
    labels = as_integer_array(
        labels,
        name="labels",
    )

    allowed = as_integer_array(
        allowed_labels,
        name="allowed_labels",
    )

    if len(allowed) == 0:
        raise ValueError("allowed_labels must contain at least one label.")

    return np.isin(labels, allowed)


def hit_count_cut(
    hit_counts: ArrayLike,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> NDArray[np.bool_]:
    """

    Select events within an inclusive hit-count range.

    Parameters
    ----------
    hit_counts : ArrayLike
        Array of hit counts for each event.
    minimum : int | None, optional
        Minimum hit count (inclusive). If None, no minimum is applied.
    maximum : int | None, optional
        Maximum hit count (inclusive). If None, no maximum is applied.

    Returns
    -------
    NDArray[np.bool_]
        Boolean mask indicating events within the specified hit-count range.
    """
    hit_counts = as_integer_array(
        hit_counts,
        name="hit_counts",
    )

    return range_cut(
        hit_counts,
        minimum=minimum,
        maximum=maximum,
        name="hit_counts",
    )


def keep_event_cut(
    keep_event: ArrayLike,
) -> NDArray[np.bool_]:
    """

    Select events marked by the input keep-event mask.


    Parameters
    ----------
    keep_event : ArrayLike
        Boolean array indicating which events to keep.

    Returns
    -------
    NDArray[np.bool_]
        Boolean mask indicating events to keep.
    """
    array = np.asarray(keep_event)

    if array.ndim > 1:
        singleton_shape = all(size == 1 for size in array.shape[1:])

        if not singleton_shape:
            raise ValueError("keep_event must contain one value per event.")

        array = array.reshape(array.shape[0])

    if array.ndim != 1:
        raise ValueError("keep_event must be one-dimensional.")

    if np.issubdtype(array.dtype, np.bool_):
        return array.copy()

    if not np.all(np.isin(array, [0, 1])):
        raise ValueError("keep_event must contain only boolean or 0/1 values.")

    return array.astype(bool)


def energy_cut(
    energies: ArrayLike,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> NDArray[np.bool_]:
    """

    Select events within an inclusive energy range.

    Parameters
    ----------
    energies : ArrayLike
        Array of energies for each event.
    minimum : float | None, optional
        Minimum energy (inclusive). If None, no minimum is applied.
    maximum : float | None, optional
        Maximum energy (inclusive). If None, no maximum is applied.

    Returns
    -------
    NDArray[np.bool_]
        Boolean mask indicating events within the specified energy range.
    """
    energies = as_scalar_array(
        energies,
        name="energies",
    ).astype(float)

    return range_cut(
        energies,
        minimum=minimum,
        maximum=maximum,
        name="energies",
    )


def momentum_cut(
    energies: ArrayLike,
    particle_labels: ArrayLike,
    *,
    energy_type: Literal["total", "kinetic"] = "total",
    minimum: float | None = None,
    maximum: float | None = None,
) -> NDArray[np.bool_]:
    """
    Select events by momentum calculated from energy and particle label.

    Particle labels are currently:

    - 0: muon
    - 1: electron
    - 2: charged pion
    """
    energies = as_scalar_array(
        energies,
        name="energies",
    ).astype(float)
    particle_labels = as_integer_array(
        particle_labels,
        name="particle_labels",
    )

    validate_same_length(
        energies,
        particle_labels,
        names=("energies", "particle_labels"),
    )

    if energy_type == "total":
        momentum = energy_to_momentum(
            energies,
            particle_labels,
        )
    elif energy_type == "kinetic":
        momentum = kinetic_energy_to_momentum(
            energies,
            particle_labels,
        )
    else:
        raise ValueError("energy_type must be 'total' or 'kinetic'.")

    return range_cut(
        momentum,
        minimum=minimum,
        maximum=maximum,
        name="momentum",
    )


def wall_cut(
    positions: ArrayLike,
    *,
    detector_radius: float,
    detector_half_height: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> NDArray[np.bool_]:
    """

    Select events by signed distance to the nearest detector wall.

    Parameters
    ----------
    positions : ArrayLike
        Array of event positions in 3D space.
    detector_radius : float
        Radius of the cylindrical detector.
    detector_half_height : float
        Half-height of the cylindrical detector.
    minimum : float | None, optional
        Minimum distance to the wall (inclusive). If None, no minimum is applied.
    maximum : float | None, optional
        Maximum distance to the wall (inclusive). If None, no maximum is applied.


    Returns
    -------
    NDArray[np.bool_]
        Boolean mask indicating events within the specified distance range.
    """
    positions = as_vector_array(
        positions,
        size=3,
        name="positions",
    ).astype(float)

    wall = compute_wall(
        positions,
        radius=detector_radius,
        half_height=detector_half_height,
    )

    return range_cut(
        wall,
        minimum=minimum,
        maximum=maximum,
        name="wall",
    )


def towall_cut(
    positions: ArrayLike,
    directions: ArrayLike,
    *,
    detector_radius: float,
    detector_half_height: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> NDArray[np.bool_]:
    """

    Select events by forward distance to the detector boundary.

    Parameters
    ----------
    positions : ArrayLike
        Array of event positions in 3D space.
    directions : ArrayLike
        Array of event direction vectors in 3D space.
    detector_radius : float
        Radius of the cylindrical detector.
    detector_half_height : float
        Half-height of the cylindrical detector.
    minimum : float | None, optional
        Minimum distance to the wall (inclusive). If None, no minimum is applied.
    maximum : float | None, optional
        Maximum distance to the wall (inclusive). If None, no maximum is applied.

    Returns
    -------
    NDArray[np.bool_]
        Boolean mask indicating events within the specified distance range.
    """
    positions = as_vector_array(
        positions,
        size=3,
        name="positions",
    ).astype(float)
    directions = as_vector_array(
        directions,
        size=3,
        name="directions",
    ).astype(float)

    validate_same_length(
        positions,
        directions,
        names=("positions", "directions"),
    )

    towall = compute_towall(
        positions,
        directions,
        radius=detector_radius,
        half_height=detector_half_height,
    )

    return range_cut(
        towall,
        minimum=minimum,
        maximum=maximum,
        name="towall",
    )


def range_cut(
    values: ArrayLike,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    name: str = "values",
) -> NDArray[np.bool_]:
    """

    Select finite values within an inclusive range.

    Parameters
    ----------
    values : ArrayLike
        Array of values to check.
    minimum : float | None, optional
        Minimum value (inclusive). If None, no minimum is applied.
    maximum : float | None, optional
        Maximum value (inclusive). If None, no maximum is applied.
    name : str, optional
        Name of the values for error messages. Default is "values".

    Returns
    -------
    NDArray[np.bool_]
        Boolean mask indicating values within the specified range.
    """
    values = as_scalar_array(
        values,
        name=name,
    ).astype(float)

    if minimum is not None and maximum is not None and maximum < minimum:
        raise ValueError(f"{name} maximum must be greater than or equal to minimum.")

    mask = np.isfinite(values)

    if minimum is not None:
        inclusive_minimum = np.nextafter(
            float(minimum),
            -np.inf,
        )
        mask &= values >= inclusive_minimum

    if maximum is not None:
        inclusive_maximum = np.nextafter(
            float(maximum),
            np.inf,
        )
        mask &= values <= inclusive_maximum

    return mask


def combine_cuts(
    *masks: ArrayLike,
) -> NDArray[np.bool_]:
    """

    Combine multiple event-selection masks using logical AND.

    Parameters
    ----------
    *masks : ArrayLike
        Variable number of event-selection masks to combine.

    Returns
    -------
    NDArray[np.bool_]
        Boolean mask indicating events that pass all selection criteria.
    """
    if not masks:
        raise ValueError("At least one selection mask is required.")

    converted: list[NDArray[np.bool_]] = []

    for mask in masks:
        array = np.asarray(mask)

        if array.ndim != 1:
            raise ValueError("Selection masks must be one-dimensional.")

        if not np.issubdtype(array.dtype, np.bool_):
            if not np.all(np.isin(array, [0, 1])):
                raise ValueError("Selection masks must contain boolean or 0/1 values.")

        converted.append(array.astype(bool))

    validate_same_length(*converted)

    return np.logical_and.reduce(converted)
