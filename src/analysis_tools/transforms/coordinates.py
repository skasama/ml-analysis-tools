import numpy as np
from numpy.typing import ArrayLike, NDArray

from analysis_tools.data import as_vector_array
from analysis_tools.transforms.geometry import normalize_vectors


def direction_from_angles(
    angles: ArrayLike,
    *,
    axis: int = 2,
) -> NDArray[np.float64]:
    """

    Convert (zenith, azimuth) angles to Cartesian unit vectors.

    Parameters
    ----------
    angles
        Array with shape (N, 2). The columns are zenith and azimuth
        angles in radians.
    axis
        Cartesian axis corresponding to the zenith direction.
        Must be 0, 1, or 2.

    Returns
    -------
    NDArray[np.float64]
        Cartesian unit vectors with shape (N, 3).
    """
    angles = _as_angle_array(
        angles,
        name="angles",
    )
    _validate_axis(axis)

    zenith = angles[:, 0]
    azimuth = angles[:, 1]

    axial = np.cos(zenith)
    transverse_radius = np.sin(zenith)

    transverse = np.column_stack(
        (
            transverse_radius * np.cos(azimuth),
            transverse_radius * np.sin(azimuth),
        )
    )

    return np.insert(
        transverse,
        axis,
        axial,
        axis=1,
    )


def angles_from_direction(
    directions: ArrayLike,
    *,
    axis: int = 2,
) -> NDArray[np.float64]:
    """Convert Cartesian direction vectors to zenith and azimuth angles.

    Parameters
    ----------
    directions
        Cartesian direction vectors with shape (N, 3).
    axis
        Cartesian axis corresponding to the zenith direction.
        Must be 0, 1, or 2.

    Returns
    -------
    NDArray[np.float64]
        Array with shape (N, 2). The columns contain zenith and
        azimuth angles in radians. Azimuth is in [-pi, pi].
    """
    _validate_axis(axis)

    directions = as_vector_array(
        directions,
        size=3,
        name="directions",
    ).astype(float)

    directions = normalize_vectors(directions)

    axial = directions[:, axis]
    transverse = np.delete(
        directions,
        axis,
        axis=1,
    )

    zenith = np.arccos(np.clip(axial, -1.0, 1.0))
    azimuth = np.arctan2(
        transverse[:, 1],
        transverse[:, 0],
    )

    return np.column_stack(
        (
            zenith,
            azimuth,
        )
    )


def cartesian_to_cylindrical(
    positions: ArrayLike,
    *,
    axis: int = 2,
) -> NDArray[np.float64]:
    """

    Convert Cartesian positions to cylindrical coordinates.

    The returned columns are radius, azimuth, and axial position.

    Parameters
    ----------
    positions
        Cartesian positions with shape (N, 3).
    axis
        Cartesian axis used as the cylinder axis.
        Must be 0, 1, or 2.

    Returns
    -------
    NDArray[np.float64]
        Cylindrical coordinates with shape (N, 3), ordered as
        (radius, azimuth, axial). Azimuth is in [-pi, pi].
    """
    _validate_axis(axis)

    positions = as_vector_array(
        positions,
        size=3,
        name="positions",
    ).astype(float)

    axial = positions[:, axis]
    transverse = np.delete(
        positions,
        axis,
        axis=1,
    )

    radius = np.linalg.norm(
        transverse,
        axis=1,
    )
    azimuth = np.arctan2(
        transverse[:, 1],
        transverse[:, 0],
    )

    return np.column_stack(
        (
            radius,
            azimuth,
            axial,
        )
    )


def cylindrical_to_cartesian(
    coordinates: ArrayLike,
    *,
    axis: int = 2,
) -> NDArray[np.float64]:
    """
    Convert cylindrical coordinates to Cartesian positions.
    The input columns must be radius, azimuth, and axial position.

    Parameters
    ----------
    coordinates
        Cylindrical coordinates with shape (N, 3), ordered as
        (radius, azimuth, axial). Angles are in radians.
    axis
        Cartesian axis used as the cylinder axis.
        Must be 0, 1, or 2.

    Returns
    -------
    NDArray[np.float64]
        Cartesian positions with shape (N, 3).
    """
    _validate_axis(axis)

    coordinates = as_vector_array(
        coordinates,
        size=3,
        name="coordinates",
    ).astype(float)

    radius = coordinates[:, 0]
    azimuth = coordinates[:, 1]
    axial = coordinates[:, 2]

    if np.any(radius < 0.0):
        raise ValueError("Cylindrical radius must be non-negative.")

    transverse = np.column_stack(
        (
            radius * np.cos(azimuth),
            radius * np.sin(azimuth),
        )
    )

    return np.insert(
        transverse,
        axis,
        axial,
        axis=1,
    )


def _as_angle_array(
    values: ArrayLike,
    *,
    name: str,
) -> NDArray[np.float64]:
    array = np.asarray(
        values,
        dtype=float,
    )

    if array.ndim == 1:
        if array.shape[0] != 2:
            raise ValueError(f"{name} must have shape (N, 2), got {array.shape}.")

        array = array.reshape(1, 2)

    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2), got {array.shape}.")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")

    return array


def _validate_axis(axis: int) -> None:
    if axis not in {0, 1, 2}:
        raise ValueError("axis must be 0, 1, or 2.")
