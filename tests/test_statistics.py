import numpy as np
import pytest

from analysis_tools.metrics.statistics import (
    bias_magnitude,
    bias_vector,
    median,
    quantile,
)


def test_bias_vector() -> None:
    residuals = np.array(
        [
            [1.0, 0.0, -1.0],
            [1.0, 2.0, 1.0],
        ]
    )

    actual = bias_vector(residuals)

    np.testing.assert_allclose(actual, [1.0, 1.0, 0.0])


def test_bias_magnitude() -> None:
    residuals = np.array(
        [
            [1.0, 0.0, -1.0],
            [1.0, 2.0, 1.0],
        ]
    )

    actual = bias_magnitude(residuals)

    assert actual == pytest.approx(np.sqrt(2.0))


def test_median() -> None:
    actual = median([1.0, 2.0, 10.0])

    assert actual == 2.0


def test_median_ignores_non_finite_values() -> None:
    actual = median([1.0, np.nan, 3.0, np.inf])

    assert actual == 2.0


def test_quantile() -> None:
    actual = quantile(
        [1.0, 2.0, 3.0, 4.0],
        0.5,
    )

    assert actual == 2.5


def test_quantile_rejects_invalid_q() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        quantile([1.0, 2.0], 1.5)


def test_median_rejects_empty_finite_values() -> None:
    with pytest.raises(ValueError, match="finite value"):
        median([np.nan, np.inf])
