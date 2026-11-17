import numpy as np
from numpy.typing import ArrayLike, NDArray

from analysis_tools.data.arrays import (
    as_scalar_array,
    validate_same_length,
)
from analysis_tools.metrics.common import bin_centers


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


def binned_bias(
    x: ArrayLike,
    residuals: ArrayLike,
    edges: ArrayLike,
    *,
    minimum_events: int = 1,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
]:
    """

    Calculate the mean residual in each x bin.

    Parameters
    ----------
    x : ArrayLike
        The x-values to bin the residuals by.
    residuals : ArrayLike
        The residuals to calculate the bias from.
    edges : ArrayLike
        The bin edges to use for binning the x-values.
    minimum_events : int, optional
        The minimum number of events required in a bin to calculate the bias.
        Bins with fewer events will be assigned NaN. Default is 1.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]
        A tuple containing the bin centers, the mean residuals in each bin,
        and the number of events in each bin.
    """
    return _binned_statistic(
        x,
        residuals,
        edges,
        statistic="mean",
        minimum_events=minimum_events,
    )


def binned_absolute_quantile(
    x: ArrayLike,
    residuals: ArrayLike,
    edges: ArrayLike,
    *,
    quantile_level: float = 0.68,
    minimum_events: int = 1,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
]:
    """Calculate an absolute-residual quantile in each x bin."""
    if not 0.0 <= quantile_level <= 1.0:
        raise ValueError("quantile_level must be between 0 and 1.")

    return _binned_statistic(
        x,
        residuals,
        edges,
        statistic="absolute_quantile",
        quantile_level=quantile_level,
        minimum_events=minimum_events,
    )


def _binned_statistic(
    x: ArrayLike,
    values: ArrayLike,
    edges: ArrayLike,
    *,
    statistic: str,
    quantile_level: float = 0.68,
    minimum_events: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
]:
    x = as_scalar_array(x, name="x").astype(float)
    values = as_scalar_array(values, name="values").astype(float)
    edges = as_scalar_array(edges, name="edges").astype(float)

    validate_same_length(
        x,
        values,
        names=("x", "values"),
    )

    if minimum_events < 1:
        raise ValueError("minimum_events must be at least 1.")

    centers = bin_centers(edges)
    results = np.full(len(centers), np.nan, dtype=float)
    counts = np.zeros(len(centers), dtype=np.int64)

    finite_mask = np.isfinite(x) & np.isfinite(values)
    valid_x = x[finite_mask]
    valid_values = values[finite_mask]

    indices = np.digitize(valid_x, edges) - 1
    indices[valid_x == edges[-1]] = len(centers) - 1

    for index in range(len(centers)):
        selected = valid_values[indices == index]
        counts[index] = len(selected)

        if counts[index] < minimum_events:
            continue

        if statistic == "mean":
            results[index] = np.mean(selected)
        elif statistic == "absolute_quantile":
            results[index] = np.quantile(
                np.abs(selected),
                quantile_level,
            )
        else:
            raise ValueError(f"Unknown statistic: {statistic}")

    return centers, results, counts
