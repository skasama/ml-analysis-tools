from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from analysis_tools.data import as_integer_array
from analysis_tools.splitting.sampling import (
    DEFAULT_RANDOM_SEED,
)


@dataclass(frozen=True)
class SplitIndices:
    """

    Train, validation, and test event indices.

    Attributes
    ----------
    train : NDArray[np.int64]
        Event indices for the training set.
    validation : NDArray[np.int64]
        Event indices for the validation set.
    test : NDArray[np.int64]
        Event indices for the test set.
    """

    train: NDArray[np.int64]
    validation: NDArray[np.int64]
    test: NDArray[np.int64]

    @property
    def selected(self) -> NDArray[np.int64]:
        """Return all selected indices."""
        return np.concatenate(
            (
                self.train,
                self.validation,
                self.test,
            )
        )

    @property
    def train_count(self) -> int:
        return len(self.train)

    @property
    def validation_count(self) -> int:
        return len(self.validation)

    @property
    def test_count(self) -> int:
        return len(self.test)


def split_indices(
    selected_indices: ArrayLike,
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int | None = None,
    shuffle: bool = True,
) -> SplitIndices:
    """Split event indices into train, validation, and test sets.

    The test fraction is the remainder after subtracting the train and
    validation fractions.
    """
    indices = _validate_selected_indices(selected_indices)

    _validate_split_fractions(
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )

    if seed is None:
        seed = DEFAULT_RANDOM_SEED

    if shuffle:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(indices)
    else:
        indices = indices.copy()

    event_count = len(indices)

    train_count = int(np.floor(event_count * train_fraction))
    validation_count = int(np.floor(event_count * validation_fraction))

    validation_end = train_count + validation_count

    result = SplitIndices(
        train=indices[:train_count],
        validation=indices[train_count:validation_end],
        test=indices[validation_end:],
    )

    validate_split(
        result,
        expected_indices=selected_indices,
    )

    return result


def validate_split(
    split: SplitIndices,
    *,
    expected_indices: ArrayLike | None = None,
) -> None:
    """

    Validate uniqueness, overlap, and event coverage.

    Parameters
    ----------
    split : SplitIndices
        SplitIndices object containing train, validation, and test indices.
    expected_indices : ArrayLike | None, optional
        Expected selected indices. If provided, the function checks that the
        split contains the same indices. If None, no check is performed.

    Raises
    ------
    ValueError
        If the split is invalid or does not match the expected indices.
    """
    train = _validate_split_array(
        split.train,
        name="train",
    )
    validation = _validate_split_array(
        split.validation,
        name="validation",
    )
    test = _validate_split_array(
        split.test,
        name="test",
    )

    _validate_no_overlap(
        train,
        validation,
        first_name="train",
        second_name="validation",
    )
    _validate_no_overlap(
        train,
        test,
        first_name="train",
        second_name="test",
    )
    _validate_no_overlap(
        validation,
        test,
        first_name="validation",
        second_name="test",
    )

    if expected_indices is None:
        return

    expected = _validate_selected_indices(expected_indices)
    actual = np.concatenate((train, validation, test))

    if len(actual) != len(expected):
        raise ValueError(
            "Split does not contain the expected number of events: "
            f"actual={len(actual)}, expected={len(expected)}."
        )

    if not np.array_equal(
        np.sort(actual),
        np.sort(expected),
    ):
        raise ValueError("Split indices do not match the expected selected indices.")


def _validate_selected_indices(
    selected_indices: ArrayLike,
) -> NDArray[np.int64]:
    indices = as_integer_array(
        selected_indices,
        name="selected_indices",
    )

    if len(indices) == 0:
        raise ValueError("selected_indices must not be empty.")

    if np.any(indices < 0):
        raise ValueError("selected_indices must be non-negative.")

    if len(np.unique(indices)) != len(indices):
        raise ValueError("selected_indices must be unique.")

    return indices


def _validate_split_array(
    indices: ArrayLike,
    *,
    name: str,
) -> NDArray[np.int64]:
    array = as_integer_array(
        indices,
        name=name,
    )

    if np.any(array < 0):
        raise ValueError(f"{name} indices must be non-negative.")

    if len(np.unique(array)) != len(array):
        raise ValueError(f"{name} indices must be unique.")

    return array


def _validate_split_fractions(
    *,
    train_fraction: float,
    validation_fraction: float,
) -> None:
    if not 0.0 <= train_fraction <= 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")

    if not 0.0 <= validation_fraction <= 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    if train_fraction + validation_fraction > 1.0:
        raise ValueError(
            "train_fraction and validation_fraction must sum to at most 1."
        )


def _validate_no_overlap(
    first: NDArray[np.int64],
    second: NDArray[np.int64],
    *,
    first_name: str,
    second_name: str,
) -> None:
    overlap = np.intersect1d(
        first,
        second,
    )

    if overlap.size > 0:
        raise ValueError(
            f"{first_name} and {second_name} overlap: {overlap[:5].tolist()}"
        )
