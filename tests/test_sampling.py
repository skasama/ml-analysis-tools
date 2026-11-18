import numpy as np
import pytest

from analysis_tools.splitting import (
    balanced_sample_indices,
    sample_indices,
)


def test_sample_indices() -> None:
    sampled = sample_indices(
        np.arange(10),
        sample_size=4,
        seed=42,
    )

    assert len(sampled) == 4
    assert len(np.unique(sampled)) == 4
    assert np.all(np.isin(sampled, np.arange(10)))


def test_sample_indices_is_reproducible() -> None:
    first = sample_indices(
        np.arange(20),
        sample_size=5,
        seed=42,
    )
    second = sample_indices(
        np.arange(20),
        sample_size=5,
        seed=42,
    )

    np.testing.assert_array_equal(first, second)


def test_sample_indices_returns_all_candidates() -> None:
    candidates = np.arange(10)

    sampled = sample_indices(
        candidates,
        sample_size=None,
        seed=42,
    )

    np.testing.assert_array_equal(
        np.sort(sampled),
        candidates,
    )


def test_sample_indices_rejects_excessive_size() -> None:
    with pytest.raises(
        ValueError,
        match="exceeds",
    ):
        sample_indices(
            np.arange(5),
            sample_size=6,
        )


def test_balanced_sample_indices() -> None:
    candidates = np.arange(12)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

    sampled = balanced_sample_indices(
        candidates,
        labels,
        balanced_labels=[0, 1, 2],
        sample_size=9,
        seed=42,
    )

    sampled_labels = labels[sampled]

    assert len(sampled) == 9
    assert np.count_nonzero(sampled_labels == 0) == 3
    assert np.count_nonzero(sampled_labels == 1) == 3
    assert np.count_nonzero(sampled_labels == 2) == 3


def test_balanced_sample_distributes_remainder() -> None:
    candidates = np.arange(12)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

    sampled = balanced_sample_indices(
        candidates,
        labels,
        balanced_labels=[0, 1, 2],
        sample_size=8,
        seed=42,
    )

    sampled_labels = labels[sampled]

    assert np.count_nonzero(sampled_labels == 0) == 3
    assert np.count_nonzero(sampled_labels == 1) == 3
    assert np.count_nonzero(sampled_labels == 2) == 2


def test_balanced_sample_uses_largest_equal_sample() -> None:
    candidates = np.arange(9)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2])

    sampled = balanced_sample_indices(
        candidates,
        labels,
        balanced_labels=[0, 1, 2],
        sample_size=None,
        seed=42,
    )

    sampled_labels = labels[sampled]

    assert len(sampled) == 6
    assert np.count_nonzero(sampled_labels == 0) == 2
    assert np.count_nonzero(sampled_labels == 1) == 2
    assert np.count_nonzero(sampled_labels == 2) == 2


def test_balanced_sample_rejects_missing_label() -> None:
    with pytest.raises(
        ValueError,
        match="No candidate events",
    ):
        balanced_sample_indices(
            candidate_indices=[0, 1, 2],
            labels=[0, 0, 1],
            balanced_labels=[0, 1, 2],
            sample_size=3,
        )


def test_balanced_sample_rejects_insufficient_events() -> None:
    with pytest.raises(
        ValueError,
        match="Insufficient candidate events",
    ):
        balanced_sample_indices(
            candidate_indices=np.arange(5),
            labels=[0, 0, 0, 1, 1],
            balanced_labels=[0, 1],
            sample_size=6,
        )


def test_sampling_rejects_duplicate_candidates() -> None:
    with pytest.raises(
        ValueError,
        match="unique",
    ):
        sample_indices(
            [0, 1, 1],
            sample_size=2,
        )


def test_default_seed_is_reproducible() -> None:
    first = sample_indices(
        np.arange(20),
        sample_size=5,
    )
    second = sample_indices(
        np.arange(20),
        sample_size=5,
    )

    np.testing.assert_array_equal(first, second)


def test_balanced_sampling_default_seed_is_reproducible() -> None:
    candidates = np.arange(12)
    labels = np.repeat([0, 1, 2], 4)

    first = balanced_sample_indices(
        candidates,
        labels,
        balanced_labels=[0, 1, 2],
        sample_size=9,
    )
    second = balanced_sample_indices(
        candidates,
        labels,
        balanced_labels=[0, 1, 2],
        sample_size=9,
    )

    np.testing.assert_array_equal(first, second)
