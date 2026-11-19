"""Statistical uncertainty calculations."""

from statistics import NormalDist
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatResult = float | NDArray[np.float64]


def binomial_error(
    successes: ArrayLike,
    trials: ArrayLike,
) -> FloatResult:
    """

    Calculate the standard binomial error.
    The error is calculated as
        sqrt(p * (1 - p) / n),
    where p = successes / trials. Entries with zero trials return NaN.

    Parameters
    ----------
    successes
        Number of successful events.
    trials
        Total number of events.

    Returns
    -------
    FloatResult
        Standard binomial error. Entries with zero trials return NaN.
    """
    successes_array, trials_array, scalar_input = _prepare_binomial_counts(
        successes,
        trials,
    )

    probability = _binomial_probability(
        successes_array,
        trials_array,
    )

    error = np.full(
        probability.shape,
        np.nan,
        dtype=float,
    )

    valid = trials_array > 0.0

    error[valid] = np.sqrt(
        probability[valid] * (1.0 - probability[valid]) / trials_array[valid]
    )

    return _restore_scalar(
        error,
        scalar_input=scalar_input,
    )


def wilson_interval(
    successes: ArrayLike,
    trials: ArrayLike,
    *,
    confidence_level: float = 0.6827,
) -> tuple[FloatResult, FloatResult]:
    """Calculate a Wilson confidence interval.

    Parameters
    ----------
    successes
        Number of successful events.
    trials
        Total number of events.
    confidence_level
        Confidence level of the interval. The default approximately
        corresponds to one standard deviation.

    Returns
    -------
    tuple
        Lower and upper confidence bounds. Entries with zero trials
        return NaN for both bounds.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")

    successes_array, trials_array, scalar_input = _prepare_binomial_counts(
        successes,
        trials,
    )

    probability = _binomial_probability(
        successes_array,
        trials_array,
    )

    lower = np.full(
        probability.shape,
        np.nan,
        dtype=float,
    )
    upper = np.full(
        probability.shape,
        np.nan,
        dtype=float,
    )

    valid = trials_array > 0.0

    if np.any(valid):
        z_value = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
        z_squared = z_value**2

        count = trials_array[valid]
        proportion = probability[valid]

        denominator = 1.0 + z_squared / count
        center = (proportion + z_squared / (2.0 * count)) / denominator

        half_width = (
            z_value
            * np.sqrt(
                proportion * (1.0 - proportion) / count + z_squared / (4.0 * count**2)
            )
            / denominator
        )

        lower[valid] = np.clip(
            center - half_width,
            0.0,
            1.0,
        )
        upper[valid] = np.clip(
            center + half_width,
            0.0,
            1.0,
        )

    return (
        _restore_scalar(
            lower,
            scalar_input=scalar_input,
        ),
        _restore_scalar(
            upper,
            scalar_input=scalar_input,
        ),
    )


def binomial_errors(
    successes: ArrayLike,
    trials: ArrayLike,
    *,
    method: Literal["standard", "wilson"] = "standard",
    confidence_level: float = 0.6827,
) -> tuple[FloatResult, FloatResult]:
    """

    Return lower and upper binomial error widths.
    For ``method="standard"``, symmetric standard errors are returned.
    For ``method="wilson"``, asymmetric error widths are calculated
    from the Wilson interval.
    The returned values are error widths rather than interval bounds,
    so they can be passed directly to Matplotlib as ``yerr``.

    Parameters
    ----------
    successes
        Number of successful events.
    trials
        Total number of events.
    method
        Method for calculating the errors. Can be either "standard" or "wilson".
    confidence_level
        Confidence level of the Wilson interval. Only used if ``method="wilson"``.

    Returns
    -------
    tuple
        Lower and upper binomial error widths. Entries with zero trials
        return NaN for both widths.
    """
    successes_array, trials_array, scalar_input = _prepare_binomial_counts(
        successes,
        trials,
    )

    if method == "standard":
        error = np.asarray(
            binomial_error(
                successes_array,
                trials_array,
            ),
            dtype=float,
        )

        lower_error = error.copy()
        upper_error = error.copy()

    elif method == "wilson":
        probability = _binomial_probability(
            successes_array,
            trials_array,
        )

        lower_bound, upper_bound = wilson_interval(
            successes_array,
            trials_array,
            confidence_level=confidence_level,
        )

        lower_error = probability - np.asarray(
            lower_bound,
            dtype=float,
        )
        upper_error = (
            np.asarray(
                upper_bound,
                dtype=float,
            )
            - probability
        )

    else:
        raise ValueError("method must be 'standard' or 'wilson'.")

    return (
        _restore_scalar(
            lower_error,
            scalar_input=scalar_input,
        ),
        _restore_scalar(
            upper_error,
            scalar_input=scalar_input,
        ),
    )


def _prepare_binomial_counts(
    successes: ArrayLike,
    trials: ArrayLike,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    bool,
]:
    successes_array = np.asarray(
        successes,
        dtype=float,
    )
    trials_array = np.asarray(
        trials,
        dtype=float,
    )

    scalar_input = successes_array.ndim == 0 and trials_array.ndim == 0

    try:
        successes_array, trials_array = np.broadcast_arrays(
            successes_array,
            trials_array,
        )
    except ValueError as error:
        raise ValueError(
            "successes and trials are not broadcast-compatible."
        ) from error

    successes_array = successes_array.astype(
        float,
        copy=False,
    )
    trials_array = trials_array.astype(
        float,
        copy=False,
    )

    if not np.all(np.isfinite(successes_array)):
        raise ValueError("successes must contain only finite values.")

    if not np.all(np.isfinite(trials_array)):
        raise ValueError("trials must contain only finite values.")

    if np.any(successes_array < 0.0):
        raise ValueError("successes must be non-negative.")

    if np.any(trials_array < 0.0):
        raise ValueError("trials must be non-negative.")

    if np.any(successes_array > trials_array):
        raise ValueError("successes must not exceed trials.")

    if not np.all(successes_array == np.floor(successes_array)):
        raise ValueError("successes must contain integer counts.")

    if not np.all(trials_array == np.floor(trials_array)):
        raise ValueError("trials must contain integer counts.")

    return (
        successes_array,
        trials_array,
        scalar_input,
    )


def _binomial_probability(
    successes: NDArray[np.float64],
    trials: NDArray[np.float64],
) -> NDArray[np.float64]:
    probability = np.full(
        successes.shape,
        np.nan,
        dtype=float,
    )

    valid = trials > 0.0

    probability[valid] = successes[valid] / trials[valid]

    return probability


def _restore_scalar(
    values: NDArray[np.float64],
    *,
    scalar_input: bool,
) -> FloatResult:
    if scalar_input:
        return float(values)

    return values
