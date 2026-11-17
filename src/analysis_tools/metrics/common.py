import numpy as np
from numpy.typing import ArrayLike, NDArray

from analysis_tools.data.arrays import (
    as_scalar_array,
    validate_same_length,
)


def make_bin_edges(
    minimum: float,
    maximum: float,
    width: float,
) -> NDArray[np.float64]:
    """

    Create evenly spaced bin edges.

    Parameters
    ----------
    minimum: float
        The minimum value of the bin edges.
    maximum: float
        The maximum value of the bin edges.
    width: float
        The width of each bin.

    Returns
    -------
    NDArray[np.float64]
        An array of bin edges from minimum to maximum with the specified width.
    """
    if maximum <= minimum:
        raise ValueError("maximum must be greater than minimum.")
    if width <= 0:
        raise ValueError("width must be positive.")

    count = int(np.ceil((maximum - minimum) / width))

    return minimum + np.arange(count + 1, dtype=float) * width


def bin_centers(edges: ArrayLike) -> NDArray[np.float64]:
    """Calculate bin centers from bin edges."""
    edges = as_scalar_array(edges, name="edges").astype(float)

    if len(edges) < 2:
        raise ValueError("At least two bin edges are required.")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("Bin edges must be strictly increasing.")

    return 0.5 * (edges[:-1] + edges[1:])


def binned_quantile(
    x: ArrayLike,
    values: ArrayLike,
    edges: ArrayLike,
    *,
    quantile: float = 0.68,
    minimum_events: int = 1,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
]:
    """

    Caluculate a quntitle of values in each x bin.

    Parameters
    ----------
    x: ArrayLike
        The x values to be binned.
    values: ArrayLike
        The values to be binned.
    edges: ArrayLike
        The bin edges.
    quantile: float
        The quantile to calculate.
    minimum_events: int
        The minimum number of events required in a bin to calculate the quantile.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]
        A tuple of the bin centers, the calculated quantiles, and the number of events
        in each bin.

    Raises
    ------
    ValueError
        If quantile is not between 0 and 1, or if minimum_events is less than 1,
        or if x and values have different lengths.
    """
    x = as_scalar_array(x, name="x").astype(float)
    values = as_scalar_array(values, name="values").astype(float)

    validate_same_length(x, values, names=("x", "values"))

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1.")
    if minimum_events < 1:
        raise ValueError("minimum_events must be at least 1.")

    centers = bin_centers(edges)
    results = np.full(len(centers), np.nan)
    counts = np.zeros(len(centers), dtype=np.int64)

    bin_indices = np.digitize(x, edges) - 1

    # include values exactly equal to the final edge in the final bin.
    bin_indices[x == edges[-1]] = len(centers) - 1

    for index in range(len(centers)):
        selected = values[bin_indices == index]
        selected = selected[np.isfinite(selected)]
        counts[index] = len(selected)
        if counts[index] >= minimum_events:
            results[index] = np.quantile(selected, quantile)

    return centers, results, counts
