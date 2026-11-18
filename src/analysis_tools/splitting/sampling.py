from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from analysis_tools.data import (
    as_integer_array,
    validate_same_length,
)

DEFAULT_RANDOM_SEED = 4242


def sample_indices(
    candidate_indices: ArrayLike,
    *,
    sample_size: int | None = None,
    seed: int | None = None,
) -> NDArray[np.int64]:
    """

    Randomly sample event indices without replacement.

    Parameters
    ----------
    candidate_indices : ArrayLike
        Array of candidate event indices to sample from.
    sample_size : int | None, optional
        Number of event indices to sample. If None, all candidate indices are returned.
    seed : int | None, optional
        Random seed for reproducibility. If None, a random seed is used.

    Returns
    -------
    NDArray[np.int64]
        Array of sampled event indices.
    """
    indices = _validate_candidate_indices(candidate_indices)

    size = _validate_sample_size(
        sample_size,
        candidate_count=len(indices),
    )

    if seed is None:
        seed = DEFAULT_RANDOM_SEED  # Default seed for reproducibility
    rng = np.random.default_rng(seed)

    if size == len(indices):
        return rng.permutation(indices)

    return rng.choice(
        indices,
        size=size,
        replace=False,
    )


def balanced_sample_indices(
    candidate_indices: ArrayLike,
    labels: ArrayLike,
    *,
    balanced_labels: Sequence[int],
    sample_size: int | None = None,
    seed: int | None = None,
) -> NDArray[np.int64]:
    """

    Sample approximately equal numbers of events from each label.

    If sample_size is not divisible by the number of labels, the
    remainder is assigned one event at a time in balanced_labels order.

    Parameters
    ----------
    candidate_indices : ArrayLike
        Array of candidate event indices to sample from.
    labels : ArrayLike
        Array of labels corresponding to candidate_indices.
    balanced_labels : Sequence[int]
        Sequence of unique labels to balance the sampling across.
    sample_size : int | None, optional
        Total number of event indices to sample. If None,
        the maximum possible balanced sample is returned.
    seed : int | None, optional
        Random seed for reproducibility. If None, a default seed (4242) is used.

    Returns
    -------
    NDArray[np.int64]
        Array of sampled event indices, balanced across the specified labels.
    """
    indices = _validate_candidate_indices(candidate_indices)
    labels = as_integer_array(
        labels,
        name="labels",
    )
    balanced_labels_array = as_integer_array(
        balanced_labels,
        name="balanced_labels",
    )

    validate_same_length(
        indices,
        labels,
        names=("candidate_indices", "labels"),
    )

    if len(balanced_labels_array) == 0:
        raise ValueError("balanced_labels must contain at least one label.")

    if len(np.unique(balanced_labels_array)) != len(balanced_labels_array):
        raise ValueError("balanced_labels must contain unique labels.")

    available_counts = np.array(
        [np.count_nonzero(labels == label) for label in balanced_labels_array],
        dtype=np.int64,
    )

    missing_labels = balanced_labels_array[available_counts == 0]

    if len(missing_labels) > 0:
        raise ValueError(
            f"No candidate events are available for labels: {missing_labels.tolist()}"
        )

    if sample_size is None:
        events_per_label = int(np.min(available_counts))
        requested_counts = np.full(
            len(balanced_labels_array),
            events_per_label,
            dtype=np.int64,
        )
    else:
        if sample_size < 1:
            raise ValueError("sample_size must be positive.")

        base_count, remainder = divmod(
            sample_size,
            len(balanced_labels_array),
        )

        requested_counts = np.full(
            len(balanced_labels_array),
            base_count,
            dtype=np.int64,
        )
        requested_counts[:remainder] += 1

    unavailable = requested_counts > available_counts

    if np.any(unavailable):
        details = ", ".join(
            (
                f"label {int(label)}: "
                f"requested={int(requested)}, "
                f"available={int(available)}"
            )
            for label, requested, available in zip(
                balanced_labels_array[unavailable],
                requested_counts[unavailable],
                available_counts[unavailable],
                strict=True,
            )
        )

        raise ValueError(
            f"Insufficient candidate events for balanced sampling: {details}"
        )

    if seed is None:
        seed = DEFAULT_RANDOM_SEED  # Default seed for reproducibility
    rng = np.random.default_rng(seed)
    sampled_groups: list[NDArray[np.int64]] = []

    for label, requested_count in zip(
        balanced_labels_array,
        requested_counts,
        strict=True,
    ):
        label_indices = indices[labels == label]

        selected = rng.choice(
            label_indices,
            size=int(requested_count),
            replace=False,
        )
        sampled_groups.append(selected)

    sampled = np.concatenate(sampled_groups)

    return rng.permutation(sampled)


def _validate_candidate_indices(
    candidate_indices: ArrayLike,
) -> NDArray[np.int64]:
    indices = as_integer_array(
        candidate_indices,
        name="candidate_indices",
    )

    if len(indices) == 0:
        raise ValueError("candidate_indices must not be empty.")

    if np.any(indices < 0):
        raise ValueError("candidate_indices must be non-negative.")

    if len(np.unique(indices)) != len(indices):
        raise ValueError("candidate_indices must be unique.")

    return indices


def _validate_sample_size(
    sample_size: int | None,
    *,
    candidate_count: int,
) -> int:
    if sample_size is None:
        return candidate_count

    if sample_size < 1:
        raise ValueError("sample_size must be positive.")

    if sample_size > candidate_count:
        raise ValueError(
            "sample_size exceeds the number of candidate events: "
            f"requested={sample_size}, "
            f"available={candidate_count}"
        )

    return sample_size
