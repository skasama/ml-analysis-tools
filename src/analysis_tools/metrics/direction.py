import numpy as np
from numpy.typing import ArrayLike, NDArray

from analysis_tools.data.arrays import validate_same_length
from analysis_tools.transforms.geometry import normalize_vectors


def direction_cosines(
    predictions: ArrayLike,
    targets: ArrayLike,
) -> NDArray[np.float64]:
    """

    Calculate cosine similarities between direction vectors.

    Parameters
    ----------
    predictions: ArrayLike
        An array of shape (N, 3) representing N predicted direction vectors.
    targets: ArrayLike
        An array of shape (N, 3) representing N target direction vectors.

    Returns
    -------
    NDArray[np.float64]
        An array of shape (N,) representing the cosine similarities between
        the predicted and target direction vectors.
    """
    predictions = normalize_vectors(predictions)
    targets = normalize_vectors(targets)

    validate_same_length(
        predictions,
        targets,
        names=("predictions", "targets"),
    )

    cosines = np.sum(predictions * targets, axis=1)

    return np.clip(cosines, -1.0, 1.0)


def direction_errors(
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    degrees: bool = True,
) -> NDArray[np.float64]:
    """

    Calculate opening-angle errors between direction vectors.

    Parameters
    ----------
    predictions: ArrayLike
        An array of shape (N, 3) representing N predicted direction vectors.
    targets: ArrayLike
        An array of shape (N, 3) representing N target direction vectors.
    degrees: bool, optional
        If True, return the errors in degrees. If False, return the errors
        in radians. Default is True.

    Returns
    -------
    NDArray[np.float64]
        An array of shape (N,) representing the opening-angle errors between
        the predicted and target direction vectors.
    """
    angles = np.arccos(direction_cosines(predictions, targets))

    if degrees:
        angles = np.degrees(angles)

    return angles


def wrapped_angle_residuals(
    predictions: ArrayLike,
    targets: ArrayLike,
) -> NDArray[np.float64]:
    """Return signed prediction-minus-target angle differences in [-pi, pi]."""
    prediction_array = np.asarray(predictions, dtype=float)
    target_array = np.asarray(targets, dtype=float)

    if prediction_array.ndim != 1 or target_array.ndim != 1:
        raise ValueError("predictions and targets must be one-dimensional.")

    validate_same_length(
        prediction_array,
        target_array,
        names=("predictions", "targets"),
    )

    difference = prediction_array - target_array
    return np.arctan2(np.sin(difference), np.cos(difference))
