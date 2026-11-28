from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from analysis_tools.data.binning import make_binned_selection

Statistic = Callable[[NDArray[np.float64]], Any]


@dataclass(frozen=True)
class MetricResult:
    """One scalar metric and optional uncertainty."""

    value: float
    error: float | tuple[float, float] | None = None


@dataclass(frozen=True)
class DependencyEvaluation:
    """A metric evaluated in bins of a dependency variable."""

    edges: NDArray[np.float64]
    centers: NDArray[np.float64]
    values: NDArray[np.float64]
    errors: NDArray[np.float64] | None
    counts: NDArray[np.int64]

    @property
    def number_of_bins(self) -> int:
        return len(self.centers)


def evaluate(
    values: ArrayLike,
    *,
    metric: Statistic = np.mean,
    error: Statistic | None = None,
) -> MetricResult:
    """Evaluate one explicit metric over event values."""
    array = np.asarray(values, dtype=float)

    if array.ndim != 1:
        raise ValueError("values must be one-dimensional.")

    finite = array[np.isfinite(array)]

    if len(finite) == 0:
        raise ValueError("values contain no finite events.")

    metric_value = _scalar_result(
        metric(finite),
        name="metric",
    )

    metric_error = None

    if error is not None:
        metric_error = _metric_error(
            error(finite),
        )

    return MetricResult(
        value=metric_value,
        error=metric_error,
    )


def evaluate_dependency(
    dependency: ArrayLike,
    values: ArrayLike,
    edges: ArrayLike,
    *,
    metric: Statistic = np.mean,
    error: Statistic | None = None,
    minimum_events: int = 1,
) -> DependencyEvaluation:
    """Evaluate a metric separately in dependency bins."""
    dependency_array = np.asarray(dependency, dtype=float)
    value_array = np.asarray(values, dtype=float)

    if dependency_array.ndim != 1:
        raise ValueError("dependency must be one-dimensional.")

    if value_array.ndim != 1:
        raise ValueError("values must be one-dimensional.")

    if len(dependency_array) != len(value_array):
        raise ValueError("dependency and values must have the same number of events.")

    if minimum_events < 1:
        raise ValueError("minimum_events must be at least 1.")

    binned = make_binned_selection(
        dependency_array,
        edges,
    )

    result_values = np.full(
        binned.number_of_bins,
        np.nan,
        dtype=float,
    )

    result_errors: NDArray[np.float64] | None = None

    if error is not None:
        result_errors = np.full(
            binned.number_of_bins,
            np.nan,
            dtype=float,
        )

    asymmetric_error = False
    asymmetric_errors: NDArray[np.float64] | None = None
    error_shape: str | None = None
    evaluated_counts = np.zeros(
        binned.number_of_bins,
        dtype=np.int64,
    )

    for bin_index in range(binned.number_of_bins):
        selected_mask = binned.mask(bin_index)
        selected_values = value_array[selected_mask]
        selected_values = selected_values[np.isfinite(selected_values)]
        evaluated_counts[bin_index] = len(selected_values)

        if len(selected_values) < minimum_events:
            continue

        result_values[bin_index] = _scalar_result(
            metric(selected_values),
            name="metric",
        )

        if error is not None:
            calculated_error = _metric_error(error(selected_values))
            calculated_shape = (
                "asymmetric" if isinstance(calculated_error, tuple) else "symmetric"
            )

            if error_shape is not None and error_shape != calculated_shape:
                raise ValueError("error must return the same shape for every bin.")

            error_shape = calculated_shape

            if isinstance(calculated_error, tuple):
                asymmetric_error = True

                if asymmetric_errors is None:
                    asymmetric_errors = np.full(
                        (2, binned.number_of_bins),
                        np.nan,
                        dtype=float,
                    )

                asymmetric_errors[0, bin_index] = calculated_error[0]
                asymmetric_errors[1, bin_index] = calculated_error[1]
            else:
                if result_errors is None:
                    result_errors = np.full(
                        binned.number_of_bins,
                        np.nan,
                        dtype=float,
                    )

                result_errors[bin_index] = calculated_error

    if asymmetric_error:
        result_errors = asymmetric_errors

    return DependencyEvaluation(
        edges=binned.edges,
        centers=binned.centers,
        values=result_values,
        errors=result_errors,
        counts=evaluated_counts,
    )


def _scalar_result(
    result: Any,
    *,
    name: str,
) -> float:
    array = np.asarray(result)

    if array.ndim != 0:
        raise ValueError(f"{name} must return one scalar value.")

    value = float(array)

    if not np.isfinite(value):
        return np.nan

    return value


def _metric_error(
    result: Any,
) -> float | tuple[float, float]:
    array = np.asarray(result)

    if array.shape == (2,):
        lower = float(array[0])
        upper = float(array[1])

        if lower < 0.0 or upper < 0.0:
            raise ValueError("asymmetric errors must be non-negative.")
        return lower, upper

    if array.ndim != 0:
        raise ValueError("error must return a scalar or a pair of scalars.")

    value = float(array)

    if value < 0.0:
        raise ValueError("error must be non-negative.")

    return value
