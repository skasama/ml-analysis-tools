from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares
from numpy.typing import ArrayLike, NDArray

from analysis_tools.fitting.result import (
    AsymmetricError,
    FitResult,
)

ModelFunction = Callable[..., ArrayLike]

ParameterLimits = Mapping[
    str,
    tuple[float | None, float | None],
]


def fit_least_squares(
    model: ModelFunction,
    x: ArrayLike,
    y: ArrayLike,
    yerr: ArrayLike,
    initial: Mapping[str, float],
    *,
    limits: ParameterLimits | None = None,
    fixed: Mapping[str, bool] | None = None,
    run_hesse: bool = True,
    run_minos: bool = False,
    minos_parameters: Sequence[str] | None = None,
    strategy: int = 1,
    tolerance: float | None = None,
    max_calls: int | None = None,
) -> FitResult:
    """Fit a model using Minuit and a least-squares cost function.

    Parameters
    ----------
    model
        Function with signature ``model(x, parameter_1, ...)``.
    x
        Independent-variable values.
    y
        Observed values.
    yerr
        Standard deviation of each observed value.
    initial
        Initial parameter values, keyed by parameter name.
    limits
        Optional lower and upper parameter limits.
    fixed
        Optional mapping indicating fixed parameters.
    run_hesse
        Run HESSE after MIGRAD.
    run_minos
        Run MINOS after MIGRAD and HESSE.
    minos_parameters
        Parameters for MINOS. All free parameters are used when omitted.
    strategy
        Minuit strategy level: 0, 1, or 2.
    tolerance
        Optional Minuit convergence tolerance.
    max_calls
        Optional maximum number of function evaluations.

    Returns
    -------
    FitResult
        Result of the fit.
    """
    x_array = np.asarray(
        x,
        dtype=float,
    )
    y_array = _as_one_dimensional(
        y,
        name="y",
    )
    yerr_array = _as_one_dimensional(
        yerr,
        name="yerr",
    )

    if len(y_array) != len(yerr_array):
        raise ValueError("y and yerr must have the same length.")

    if y_array.size == 0:
        raise ValueError("At least one data point is required.")

    if not np.all(np.isfinite(x_array)):
        raise ValueError("x must contain only finite values.")

    if not np.all(np.isfinite(y_array)):
        raise ValueError("y must contain only finite values.")

    if not np.all(np.isfinite(yerr_array)):
        raise ValueError("yerr must contain only finite values.")

    if np.any(yerr_array <= 0.0):
        raise ValueError("yerr must contain only positive values.")

    if not initial:
        raise ValueError("At least one initial parameter is required.")

    cost = LeastSquares(
        x_array,
        y_array,
        yerr_array,
        model,
    )

    minuit = Minuit(
        cost,
        **dict(initial),
    )

    _configure_minuit(
        minuit,
        limits=limits,
        fixed=fixed,
        strategy=strategy,
        tolerance=tolerance,
    )

    minuit.migrad(
        ncall=max_calls,
    )

    if run_hesse:
        minuit.hesse()

    if run_minos:
        if minos_parameters is None:
            minuit.minos()
        else:
            _validate_parameter_names(
                minos_parameters,
                minuit.parameters,
                setting_name="minos_parameters",
            )
            minuit.minos(*minos_parameters)

    return _build_result(
        minuit,
        number_of_data_points=len(y_array),
    )


def _configure_minuit(
    minuit: Minuit,
    *,
    limits: ParameterLimits | None,
    fixed: Mapping[str, bool] | None,
    strategy: int,
    tolerance: float | None,
) -> None:
    if strategy not in {0, 1, 2}:
        raise ValueError("strategy must be 0, 1, or 2.")

    minuit.strategy = strategy

    if tolerance is not None:
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive.")

        minuit.tol = tolerance

    if limits is not None:
        _validate_parameter_names(
            limits,
            minuit.parameters,
            setting_name="limits",
        )

        for name, parameter_limits in limits.items():
            lower, upper = parameter_limits

            if lower is not None and upper is not None and upper < lower:
                raise ValueError(
                    f"Invalid limits for {name}: lower={lower}, upper={upper}."
                )

            minuit.limits[name] = (
                lower,
                upper,
            )

    if fixed is not None:
        _validate_parameter_names(
            fixed,
            minuit.parameters,
            setting_name="fixed",
        )

        for name, is_fixed in fixed.items():
            minuit.fixed[name] = is_fixed


def _build_result(
    minuit: Minuit,
    *,
    number_of_data_points: int,
) -> FitResult:
    parameter_names = tuple(minuit.parameters)

    values = {name: float(minuit.values[name]) for name in parameter_names}
    errors = {name: float(minuit.errors[name]) for name in parameter_names}

    covariance = None

    if minuit.covariance is not None:
        covariance = np.asarray(
            minuit.covariance,
            dtype=float,
        ).copy()

    minos_errors = {
        name: AsymmetricError(
            lower=float(error.lower),
            upper=float(error.upper),
            valid=bool(error.lower_valid and error.upper_valid),
        )
        for name, error in minuit.merrors.items()
    }

    free_parameter_count = sum(not minuit.fixed[name] for name in parameter_names)

    fit_minimum = minuit.fmin

    return FitResult(
        parameter_names=parameter_names,
        values=values,
        errors=errors,
        covariance=covariance,
        minos_errors=minos_errors,
        minimum_value=float(minuit.fval),
        valid=bool(minuit.valid),
        accurate_covariance=bool(fit_minimum.has_accurate_covar),
        reached_call_limit=bool(fit_minimum.has_reached_call_limit),
        above_maximum_edm=bool(fit_minimum.is_above_max_edm),
        number_of_data_points=number_of_data_points,
        number_of_free_parameters=free_parameter_count,
    )


def _validate_parameter_names(
    settings: Mapping[str, Any] | Sequence[str],
    parameter_names: Sequence[str],
    *,
    setting_name: str,
) -> None:
    names = settings.keys() if isinstance(settings, Mapping) else settings

    unknown = sorted(set(names) - set(parameter_names))

    if unknown:
        raise ValueError(f"{setting_name} contains unknown parameters: {unknown}")


def _as_one_dimensional(
    values: ArrayLike,
    *,
    name: str,
) -> NDArray[np.float64]:
    array = np.asarray(
        values,
        dtype=float,
    )

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}.")

    return array
