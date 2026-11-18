import numpy as np
import pytest

from analysis_tools.splitting import (
    SplitIndices,
    split_indices,
    validate_split,
)


def test_split_indices() -> None:
    result = split_indices(
        np.arange(10),
        train_fraction=0.6,
        validation_fraction=0.2,
        seed=42,
    )

    assert result.train_count == 6
    assert result.validation_count == 2
    assert result.test_count == 2

    np.testing.assert_array_equal(
        np.sort(result.selected),
        np.arange(10),
    )


def test_split_indices_uses_remainder_for_test() -> None:
    result = split_indices(
        np.arange(11),
        train_fraction=0.6,
        validation_fraction=0.2,
        seed=42,
    )

    assert result.train_count == 6
    assert result.validation_count == 2
    assert result.test_count == 3


def test_split_indices_default_seed_is_reproducible() -> None:
    first = split_indices(np.arange(20))
    second = split_indices(np.arange(20))

    np.testing.assert_array_equal(
        first.train,
        second.train,
    )
    np.testing.assert_array_equal(
        first.validation,
        second.validation,
    )
    np.testing.assert_array_equal(
        first.test,
        second.test,
    )


def test_split_indices_without_shuffle() -> None:
    result = split_indices(
        np.arange(10),
        train_fraction=0.6,
        validation_fraction=0.2,
        shuffle=False,
    )

    np.testing.assert_array_equal(
        result.train,
        np.arange(6),
    )
    np.testing.assert_array_equal(
        result.validation,
        [6, 7],
    )
    np.testing.assert_array_equal(
        result.test,
        [8, 9],
    )


def test_split_indices_rejects_invalid_fractions() -> None:
    with pytest.raises(
        ValueError,
        match="sum to at most 1",
    ):
        split_indices(
            np.arange(10),
            train_fraction=0.8,
            validation_fraction=0.3,
        )


def test_split_indices_rejects_duplicate_indices() -> None:
    with pytest.raises(
        ValueError,
        match="unique",
    ):
        split_indices([0, 1, 1, 2])


def test_validate_split_rejects_overlap() -> None:
    split = SplitIndices(
        train=np.array([0, 1]),
        validation=np.array([1, 2]),
        test=np.array([3]),
    )

    with pytest.raises(
        ValueError,
        match="overlap",
    ):
        validate_split(split)


def test_validate_split_rejects_missing_event() -> None:
    split = SplitIndices(
        train=np.array([0, 1]),
        validation=np.array([2]),
        test=np.array([3]),
    )

    with pytest.raises(
        ValueError,
        match="expected number",
    ):
        validate_split(
            split,
            expected_indices=np.arange(5),
        )


def test_validate_split_rejects_unexpected_event() -> None:
    split = SplitIndices(
        train=np.array([0, 1]),
        validation=np.array([2]),
        test=np.array([4]),
    )

    with pytest.raises(
        ValueError,
        match="do not match",
    ):
        validate_split(
            split,
            expected_indices=[0, 1, 2, 3],
        )
