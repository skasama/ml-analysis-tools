import numpy as np
import pytest

from analysis_tools.data import (
    hit_counts_from_indices,
)


def test_hit_counts_from_indices() -> None:
    actual = hit_counts_from_indices(
        event_hits_index=[0, 3, 8],
        total_hit_count=10,
    )

    np.testing.assert_array_equal(
        actual,
        [3, 5, 2],
    )


def test_hit_counts_supports_empty_events() -> None:
    actual = hit_counts_from_indices(
        event_hits_index=[0, 3, 3, 8],
        total_hit_count=10,
    )

    np.testing.assert_array_equal(
        actual,
        [3, 0, 5, 2],
    )


def test_hit_counts_rejects_unsorted_indices() -> None:
    with pytest.raises(
        ValueError,
        match="monotonically increasing",
    ):
        hit_counts_from_indices(
            event_hits_index=[0, 8, 3],
            total_hit_count=10,
        )


def test_hit_counts_rejects_excessive_index() -> None:
    with pytest.raises(
        ValueError,
        match="exceeds",
    ):
        hit_counts_from_indices(
            event_hits_index=[0, 11],
            total_hit_count=10,
        )
