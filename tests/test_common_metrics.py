import numpy as np
import pytest

from analysis_tools.metrics.common import (
    bin_centers,
    binned_quantile,
    make_bin_edges,
)


def test_make_bin_edges() -> None:
    actual = make_bin_edges(0.0, 10.0, 2.0)

    np.testing.assert_allclose(actual, [0, 2, 4, 6, 8, 10])


def test_make_bin_edges_extends_to_cover_maximum() -> None:
    actual = make_bin_edges(0.0, 9.0, 2.0)

    np.testing.assert_allclose(actual, [0, 2, 4, 6, 8, 10])


def test_bin_centers() -> None:
    actual = bin_centers([0.0, 2.0, 4.0])

    np.testing.assert_allclose(actual, [1.0, 3.0])


def test_binned_quantile() -> None:
    centers, values, counts = binned_quantile(
        x=[0.5, 1.5, 2.5, 3.5],
        values=[1.0, 3.0, 5.0, 7.0],
        edges=[0.0, 2.0, 4.0],
        quantile=0.5,
    )

    np.testing.assert_allclose(centers, [1.0, 3.0])
    np.testing.assert_allclose(values, [2.0, 6.0])
    np.testing.assert_array_equal(counts, [2, 2])


def test_binned_quantile_includes_final_edge() -> None:
    _, values, counts = binned_quantile(
        x=[0.0, 1.0, 2.0],
        values=[1.0, 2.0, 3.0],
        edges=[0.0, 1.0, 2.0],
        quantile=0.5,
    )

    np.testing.assert_allclose(values, [1.0, 2.5])
    np.testing.assert_array_equal(counts, [1, 2])


def test_binned_quantile_masks_low_statistics_bins() -> None:
    _, values, counts = binned_quantile(
        x=[0.5, 2.5],
        values=[1.0, 2.0],
        edges=[0.0, 2.0, 4.0],
        minimum_events=2,
    )

    assert np.all(np.isnan(values))
    np.testing.assert_array_equal(counts, [1, 1])


def test_binned_quantile_rejects_invalid_quantile() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        binned_quantile(
            x=[1.0],
            values=[1.0],
            edges=[0.0, 2.0],
            quantile=1.5,
        )
