import numpy as np
import pytest

from analysis_tools.fitting import (
    fit_least_squares,
)


def linear_model(
    x: np.ndarray,
    slope: float,
    intercept: float,
) -> np.ndarray:
    return slope * x + intercept


def test_fit_least_squares_linear_model() -> None:
    x = np.linspace(
        0.0,
        10.0,
        21,
    )
    yerr = np.full_like(
        x,
        0.2,
    )
    y = linear_model(
        x,
        slope=2.0,
        intercept=1.0,
    )

    result = fit_least_squares(
        linear_model,
        x,
        y,
        yerr,
        initial={
            "slope": 1.0,
            "intercept": 0.0,
        },
    )

    assert result.valid
    assert result.value("slope") == pytest.approx(
        2.0,
        abs=1e-6,
    )
    assert result.value("intercept") == pytest.approx(
        1.0,
        abs=1e-6,
    )

    assert result.error("slope") > 0.0
    assert result.error("intercept") > 0.0

    assert result.covariance is not None
    assert result.covariance.shape == (2, 2)

    assert result.number_of_data_points == 21
    assert result.number_of_free_parameters == 2
    assert result.degrees_of_freedom == 19

    assert result.minimum_value == pytest.approx(
        0.0,
        abs=1e-8,
    )


def test_fit_least_squares_with_fixed_parameter() -> None:
    x = np.linspace(
        0.0,
        10.0,
        21,
    )
    y = linear_model(
        x,
        slope=2.0,
        intercept=1.0,
    )

    result = fit_least_squares(
        linear_model,
        x,
        y,
        np.ones_like(x),
        initial={
            "slope": 1.0,
            "intercept": 1.0,
        },
        fixed={
            "intercept": True,
        },
    )

    assert result.valid
    assert result.value("slope") == pytest.approx(
        2.0,
        abs=1e-6,
    )
    assert result.value("intercept") == 1.0
    assert result.number_of_free_parameters == 1
    assert result.degrees_of_freedom == 20


def test_fit_least_squares_with_limits() -> None:
    x = np.linspace(
        0.0,
        10.0,
        21,
    )
    y = linear_model(
        x,
        slope=2.0,
        intercept=1.0,
    )

    result = fit_least_squares(
        linear_model,
        x,
        y,
        np.ones_like(x),
        initial={
            "slope": 1.0,
            "intercept": 0.0,
        },
        limits={
            "slope": (0.0, 5.0),
        },
    )

    assert result.valid
    assert 0.0 <= result.value("slope") <= 5.0
    assert result.value("slope") == pytest.approx(
        2.0,
        abs=1e-5,
    )


def test_fit_least_squares_with_minos() -> None:
    x = np.linspace(
        0.0,
        10.0,
        21,
    )
    y = linear_model(
        x,
        slope=2.0,
        intercept=1.0,
    )

    result = fit_least_squares(
        linear_model,
        x,
        y,
        np.ones_like(x),
        initial={
            "slope": 1.0,
            "intercept": 0.0,
        },
        run_minos=True,
        minos_parameters=["slope"],
    )

    assert result.valid
    assert "slope" in result.minos_errors

    minos_error = result.minos_errors["slope"]

    assert minos_error.lower < 0.0
    assert minos_error.upper > 0.0
    assert minos_error.valid


def test_fit_least_squares_rejects_nonpositive_yerr() -> None:
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        fit_least_squares(
            linear_model,
            x=[0.0, 1.0],
            y=[1.0, 3.0],
            yerr=[1.0, 0.0],
            initial={
                "slope": 1.0,
                "intercept": 0.0,
            },
        )


def test_fit_least_squares_rejects_unknown_limit_parameter() -> None:
    with pytest.raises(
        ValueError,
        match="unknown parameters",
    ):
        fit_least_squares(
            linear_model,
            x=[0.0, 1.0],
            y=[1.0, 3.0],
            yerr=[1.0, 1.0],
            initial={
                "slope": 1.0,
                "intercept": 0.0,
            },
            limits={
                "unknown": (0.0, 1.0),
            },
        )


def test_fit_least_squares_rejects_invalid_limits() -> None:
    with pytest.raises(
        ValueError,
        match="Invalid limits",
    ):
        fit_least_squares(
            linear_model,
            x=[0.0, 1.0],
            y=[1.0, 3.0],
            yerr=[1.0, 1.0],
            initial={
                "slope": 1.0,
                "intercept": 0.0,
            },
            limits={
                "slope": (5.0, 0.0),
            },
        )
