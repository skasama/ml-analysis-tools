import numpy as np
import pytest

from analysis_tools.data import (
    BinnedSelection,
    make_binned_selection,
)


def test_make_binned_selection() -> None:
    bins = make_binned_selection(
        values=[0.0, 0.5, 1.0, 1.5, 2.0],
        edges=[0.0, 1.0, 2.0],
    )

    assert isinstance(
        bins,
        BinnedSelection,
    )
    assert bins.number_of_bins == 2
    assert bins.event_count == 5

    np.testing.assert_allclose(
        bins.centers,
        [0.5, 1.5],
    )
    np.testing.assert_array_equal(
        bins.counts,
        [2, 3],
    )
    np.testing.assert_array_equal(
        bins.indices(0),
        [0, 1],
    )
    np.testing.assert_array_equal(
        bins.indices(1),
        [2, 3, 4],
    )


def test_make_binned_selection_excludes_right_edge() -> None:
    bins = make_binned_selection(
        values=[0.0, 1.0, 2.0],
        edges=[0.0, 1.0, 2.0],
        include_right_edge=False,
    )

    np.testing.assert_array_equal(
        bins.counts,
        [1, 1],
    )


def test_make_binned_selection_excludes_nonfinite_values() -> None:
    bins = make_binned_selection(
        values=[0.5, np.nan, np.inf, 1.5],
        edges=[0.0, 1.0, 2.0],
    )

    np.testing.assert_array_equal(
        bins.counts,
        [1, 1],
    )


def test_binned_selection_mask_returns_copy() -> None:
    bins = make_binned_selection(
        values=[0.5, 1.5],
        edges=[0.0, 1.0, 2.0],
    )

    mask = bins.mask(0)
    mask[0] = False

    np.testing.assert_array_equal(
        bins.mask(0),
        [True, False],
    )


def test_binned_selection_valid_bins() -> None:
    bins = make_binned_selection(
        values=[0.2, 0.4, 1.5],
        edges=[0.0, 1.0, 2.0],
    )

    np.testing.assert_array_equal(
        bins.valid_bins(minimum_events=2),
        [True, False],
    )


def test_make_binned_selection_rejects_invalid_edges() -> None:
    with pytest.raises(
        ValueError,
        match="strictly increasing",
    ):
        make_binned_selection(
            values=[0.5],
            edges=[0.0, 1.0, 1.0],
        )


def test_make_binned_selection_rejects_multidimensional_values() -> None:
    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        make_binned_selection(
            values=[[0.5], [1.5]],
            edges=[0.0, 1.0, 2.0],
        )
