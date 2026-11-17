import numpy as np
import pytest

from analysis_tools.metrics.energy import (
    energy_residuals,
    fractional_energy_residuals,
)


def test_energy_residuals() -> None:
    actual = energy_residuals(
        predictions=[110.0, 180.0],
        targets=[100.0, 200.0],
    )

    np.testing.assert_allclose(actual, [10.0, -20.0])


def test_fractional_energy_residuals() -> None:
    actual = fractional_energy_residuals(
        predictions=[110.0, 180.0],
        targets=[100.0, 200.0],
    )

    np.testing.assert_allclose(actual, [0.1, -0.1])


def test_energy_metrics_accept_singleton_dimensions() -> None:
    actual = energy_residuals(
        predictions=[[110.0], [180.0]],
        targets=[[100.0], [200.0]],
    )

    np.testing.assert_allclose(actual, [10.0, -20.0])


def test_energy_residuals_reject_length_mismatch() -> None:
    with pytest.raises(ValueError, match="Array lengths do not match"):
        energy_residuals(
            predictions=[100.0, 200.0],
            targets=[100.0],
        )


def test_fractional_energy_residuals_returns_nan_for_zero_target() -> None:
    actual = fractional_energy_residuals(
        predictions=[10.0, 20.0],
        targets=[0.0, 10.0],
    )

    assert np.isnan(actual[0])
    np.testing.assert_allclose(actual[1], 1.0)
