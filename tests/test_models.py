import numpy as np
import pytest

from analysis_tools.data.models import EventIndex


def test_event_index_from_arrays() -> None:
    index = EventIndex.from_arrays(
        file_ids=[10, 10, 11],
        event_ids=[0, 1, 0],
    )

    assert len(index) == 3
    np.testing.assert_array_equal(index.file_ids, [10, 10, 11])
    np.testing.assert_array_equal(index.event_ids, [0, 1, 0])


def test_event_index_accepts_singleton_dimensions() -> None:
    index = EventIndex.from_arrays(
        file_ids=np.array([[10], [11]]),
        event_ids=np.array([[0], [1]]),
    )

    assert index.file_ids.shape == (2,)
    assert index.event_ids.shape == (2,)


def test_event_index_rejects_different_lengths() -> None:
    with pytest.raises(ValueError, match="Array lengths do not match"):
        EventIndex.from_arrays(
            file_ids=[10, 11],
            event_ids=[0],
        )


def test_event_index_keys() -> None:
    index = EventIndex.from_arrays(
        file_ids=[10, 10, 11],
        event_ids=[0, 1, 0],
    )

    keys = index.keys()

    np.testing.assert_array_equal(keys["file_id"], [10, 10, 11])
    np.testing.assert_array_equal(keys["event_id"], [0, 1, 0])
