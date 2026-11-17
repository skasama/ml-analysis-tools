import numpy as np
from numpy.typing import ArrayLike, NDArray


def bias_vector(
    residuals: ArrayLike,
) -> NDArray[np.float64] | np.float64:
    """

    Calculate the mean residual over the first axis.

    Parameters
    ----------
    residuals : ArrayLike
        The residuals to calculate the bias vector from. Can be a 1D or 2D array.
        If 1D, the bias vector will be a scalar.
        If 2D, the bias vector will be a 1D array with the mean of each column.

    Returns
    -------
    NDArray[np.float64] | np.float64
        The bias vector.
    """
    residuals = np.asarray(residuals, dtype=float)

    if residuals.ndim == 0:
        raise ValueError("residuals must have at least one dimension.")

    if residuals.shape[0] == 0:
        raise ValueError("residuals must contain at least one entry.")

    return np.mean(residuals, axis=0)


def bias_magnitude(
    residuals: ArrayLike,
) -> float:
    """

    Calculate the magnitude of the mean residual.

    Parameters
    ----------
    residuals : ArrayLike
        The residuals to calculate the bias magnitude from.

    Returns
    -------
    float
        The magnitude of the mean residual.
    """
    bias = np.asarray(bias_vector(residuals), dtype=float)

    return float(np.linalg.norm(bias))


def median(
    values: ArrayLike,
) -> float:
    """

    Calculate the median of finite values.

    Parameters
    ----------
    values : ArrayLike
        The values to calculate the median from.

    Returns
    -------
    float
        The median of the finite values.
    """
    values = _finite_values(values)

    return float(np.median(values))


def quantile(
    values: ArrayLike,
    q: float,
) -> float:
    """

    Calculate a quantile of finite values.

    Parameters
    ----------
    values : ArrayLike
        The values to calculate the quantile from.
    q : float
        The quantile to calculate, between 0 and 1.

    Returns
    -------
    float
        The quantile of the finite values.
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be between 0 and 1.")

    values = _finite_values(values)

    return float(np.quantile(values, q))


def _finite_values(
    values: ArrayLike,
) -> NDArray[np.float64]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError("values must contain at least one finite value.")

    return values
