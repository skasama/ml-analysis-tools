import numpy as np
import pytest

from analysis_tools.data.alignment import (
    find_duplicate_keys,
    validate_aligned,
    validate_unique_keys,
)
from analysis_tools.data.models import EventIndex


def test_find_duplicate_keys() -> None:
    index = EventIndex.from_arrays(
        file_ids=[10, 10, 10, 11],
        event_ids=[0, 1, 1, 0],
    )

    actual = find_duplicate_keys(index)

    np.testing.assert_array_equal(
        actual,
        [False, True, True, False],
    )


def test_validate_unique_keys_accepts_unique_index() -> None:
    index = EventIndex.from_arrays(
        file_ids=[10, 10, 11],
        event_ids=[0, 1, 0],
    )

    validate_unique_keys(index)


def test_validate_unique_keys_rejects_duplicates() -> None:
    index = EventIndex.from_arrays(
        file_ids=[10, 10],
        event_ids=[1, 1],
    )

    with pytest.raises(ValueError, match="Duplicate event keys"):
        validate_unique_keys(index)


def test_validate_aligned_accepts_matching_indices() -> None:
    reference = EventIndex.from_arrays([10, 10], [0, 1])
    candidate = EventIndex.from_arrays([10, 10], [0, 1])

    validate_aligned(reference, candidate)


def test_validate_aligned_rejects_different_lengths() -> None:
    reference = EventIndex.from_arrays([10, 10], [0, 1])
    candidate = EventIndex.from_arrays([10], [0])

    with pytest.raises(ValueError, match="Event counts do not match"):
        validate_aligned(reference, candidate)


def test_validate_aligned_rejects_different_order() -> None:
    reference = EventIndex.from_arrays([10, 10], [0, 1])
    candidate = EventIndex.from_arrays([10, 10], [1, 0])

    with pytest.raises(ValueError, match="position 0"):
        validate_aligned(reference, candidate)
