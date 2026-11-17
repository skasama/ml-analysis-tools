import numpy as np
from numpy.typing import ArrayLike, NDArray

from analysis_tools.data.arrays import (
    as_scalar_array,
    validate_same_length,
)


def energy_residuals(
    predictions: ArrayLike,
    targets: ArrayLike,
) -> NDArray[np.float64]:
    """

    Calculate prediction minus target

    Parameters
    ----------
    predictions : ArrayLike
        Predicted energy values.
    targets : ArrayLike
        True energy values.

    Returns
    -------
    NDArray[np.float64]
        The energy residuals, calculated as predictions - targets.
    """

    predictions = as_scalar_array(
        predictions,
        name="predictions",
    ).astype(float)
    targets = as_scalar_array(
        targets,
        name="targets",
    ).astype(float)

    validate_same_length(predictions, targets, names=("predictions", "targets"))

    return predictions - targets


def fractional_energy_residuals(
    predictions: ArrayLike,
    targets: ArrayLike,
) -> NDArray[np.float64]:
    """

    Calculate (prediction - target) / target.

    Parameters
    ----------
    predictions : ArrayLike
        Predicted energy values.
    targets : ArrayLike
        True energy values.

    Returns
    -------
    NDArray[np.float64]
        The fractional energy residuals, calculated as
        (predictions - targets) / targets.
    """
    predictions = as_scalar_array(
        predictions,
        name="predictions",
    ).astype(float)
    targets = as_scalar_array(
        targets,
        name="targets",
    ).astype(float)

    validate_same_length(predictions, targets, names=("predictions", "targets"))

    residuals = np.full(targets.shape, np.nan, dtype=float)

    np.divide(
        predictions - targets,
        targets,
        out=residuals,
        where=targets != 0,
    )

    return residuals
