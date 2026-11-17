import numpy as np
from numpy.typing import ArrayLike, NDArray

from analysis_tools.data.arrays import (
    as_vector_array,
    validate_same_length,
)


def vertex_residuals(
    predictions: ArrayLike,
    targets: ArrayLike,
) -> NDArray[np.float64]:
    """

    Calculate prediction minus target for each cordinate.

    Parameters
    ----------
    predictions: ArrayLike
        An array of shape (N, 3) representing N predicted points in 3D space.
    targets: ArrayLike
        An array of shape (N, 3) representing N target points in 3D space.

    Returns
    -------
    NDArray[np.float64]
        An array of shape (N,) representing the residuals for each point.

    Raises
    ------
    ValueError
        If the input arrays have different lengths.
    """
    predictions = as_vector_array(
        predictions,
        size=3,
        name="predictions",
    ).astype(float)
    targets = as_vector_array(
        targets,
        size=3,
        name="targets",
    ).astype(float)

    validate_same_length(
        predictions,
        targets,
        names=("predictions", "targets"),
    )

    return predictions - targets


def vertex_errors(
    predictions: ArrayLike,
    targets: ArrayLike,
) -> NDArray[np.float64]:
    """

    Calculate the three-dimensional vertex error.

    Parameters
    ----------
    predictions: ArrayLike
        An array of shape (N, 3) representing N predicted points in 3D space.
    targets: ArrayLike
        An array of shape (N, 3) representing N target points in 3D space.

    Returns
    -------
    NDArray[np.float64]
        An array of shape (N,) representing the errors for each point.
    """
    residuals = vertex_residuals(predictions, targets)

    return np.linalg.norm(residuals, axis=1)
