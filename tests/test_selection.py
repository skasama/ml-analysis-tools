import numpy as np
import pytest

from analysis_tools.data.selection import (
    apply_mask,
    apply_mask_to_mapping,
    combine_masks,
    range_mask,
)


def test_range_mask() -> None:
    actual = range_mask(
        [0.0, 10.0, 20.0, 30.0],
        minimum=10.0,
        maximum=20.0,
    )

    np.testing.assert_array_equal(
        actual,
        [False, True, True, False],
    )


def test_range_mask_exclusive() -> None:
    actual = range_mask(
        [0.0, 10.0, 20.0],
        minimum=0.0,
        maximum=20.0,
        inclusive=False,
    )

    np.testing.assert_array_equal(
        actual,
        [False, True, False],
    )


def test_combine_masks() -> None:
    actual = combine_masks(
        [True, True, False],
        [True, False, True],
    )

    np.testing.assert_array_equal(
        actual,
        [True, False, False],
    )


def test_apply_mask() -> None:
    values, labels = apply_mask(
        [True, False, True],
        [10.0, 20.0, 30.0],
        [0, 1, 2],
    )

    np.testing.assert_array_equal(values, [10.0, 30.0])
    np.testing.assert_array_equal(labels, [0, 2])


def test_apply_mask_to_mapping() -> None:
    actual = apply_mask_to_mapping(
        [True, False, True],
        {
            "values": [10.0, 20.0, 30.0],
            "labels": [0, 1, 2],
        },
    )

    np.testing.assert_array_equal(actual["values"], [10.0, 30.0])
    np.testing.assert_array_equal(actual["labels"], [0, 2])


def test_combine_masks_rejects_different_lengths() -> None:
    with pytest.raises(ValueError, match="Array lengths do not match"):
        combine_masks(
            [True, False],
            [True],
        )
