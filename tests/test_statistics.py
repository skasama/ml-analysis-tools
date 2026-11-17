import numpy as np
import pytest

from analysis_tools.metrics.statistics import (
    bias_magnitude,
    bias_vector,
    binned_absolute_quantile,
    binned_bias,
    median,
    pearson_correlation,
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


def test_binned_bias() -> None:
    centers, values, counts = binned_bias(
        x=[0.5, 1.5, 2.5, 3.5],
        residuals=[1.0, 3.0, -2.0, 2.0],
        edges=[0.0, 2.0, 4.0],
    )

    np.testing.assert_allclose(centers, [1.0, 3.0])
    np.testing.assert_allclose(values, [2.0, 0.0])
    np.testing.assert_array_equal(counts, [2, 2])


def test_binned_absolute_quantile() -> None:
    centers, values, counts = binned_absolute_quantile(
        x=[0.5, 1.5, 2.5, 3.5],
        residuals=[-1.0, 3.0, -2.0, 4.0],
        edges=[0.0, 2.0, 4.0],
        quantile_level=0.5,
    )

    np.testing.assert_allclose(centers, [1.0, 3.0])
    np.testing.assert_allclose(values, [2.0, 3.0])
    np.testing.assert_array_equal(counts, [2, 2])


def test_binned_statistics_mask_low_statistics_bins() -> None:
    _, bias, counts = binned_bias(
        x=[0.5, 2.5],
        residuals=[1.0, 2.0],
        edges=[0.0, 2.0, 4.0],
        minimum_events=2,
    )

    assert np.all(np.isnan(bias))
    np.testing.assert_array_equal(counts, [1, 1])


def test_binned_statistics_ignore_nonfinite_values() -> None:
    _, bias, counts = binned_bias(
        x=[0.5, 1.5, np.nan],
        residuals=[1.0, np.nan, 5.0],
        edges=[0.0, 2.0],
    )

    np.testing.assert_allclose(bias, [1.0])
    np.testing.assert_array_equal(counts, [1])


def test_binned_absolute_quantile_rejects_invalid_quantile() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        binned_absolute_quantile(
            x=[1.0],
            residuals=[1.0],
            edges=[0.0, 2.0],
            quantile_level=1.5,
        )


def test_pearson_correlation() -> None:
    actual = pearson_correlation(
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
    )

    assert actual == pytest.approx(1.0)


def test_pearson_correlation_negative() -> None:
    actual = pearson_correlation(
        [1.0, 2.0, 3.0],
        [6.0, 4.0, 2.0],
    )

    assert actual == pytest.approx(-1.0)


def test_pearson_correlation_ignores_nonfinite_pairs() -> None:
    actual = pearson_correlation(
        [1.0, 2.0, np.nan, 3.0],
        [2.0, 4.0, 5.0, 6.0],
    )

    assert actual == pytest.approx(1.0)


def test_pearson_correlation_rejects_different_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        pearson_correlation(
            [1.0, 2.0],
            [1.0],
        )


def test_pearson_correlation_rejects_constant_values() -> None:
    with pytest.raises(ValueError, match="constant"):
        pearson_correlation(
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 3.0],
        )


def test_pearson_correlation_requires_two_finite_pairs() -> None:
    with pytest.raises(ValueError, match="two finite pairs"):
        pearson_correlation(
            [1.0, np.nan],
            [2.0, 3.0],
        )
