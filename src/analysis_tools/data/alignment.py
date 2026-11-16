import numpy as np
from numpy.typing import NDArray

from analysis_tools.data.models import EventIndex


def find_duplicate_keys(index: EventIndex) -> NDArray[np.bool_]:
    """Return a mask identifying all entries with duplicated event keys."""
    keys = index.keys()
    _, inverse, counts = np.unique(
        keys,
        return_inverse=True,
        return_counts=True,
    )
    return counts[inverse] > 1


def validate_unique_keys(index: EventIndex) -> None:
    """Raise an error if duplicated event keys are present."""
    duplicate_mask = find_duplicate_keys(index)

    if not np.any(duplicate_mask):
        return

    duplicate_keys = np.unique(index.keys()[duplicate_mask])
    first = duplicate_keys[0]

    raise ValueError(
        f"Duplicate event keys found: {len(duplicate_keys)} unique duplicates. "
        f"First duplicate: ({first['file_id']}, {first['event_id']})."
    )


def validate_aligned(
    reference: EventIndex,
    candidate: EventIndex,
) -> None:
    """Validate that two event indices contain the same keys in the same order."""
    if len(reference) != len(candidate):
        raise ValueError(
            "Event counts do not match: "
            f"reference={len(reference)}, candidate={len(candidate)}."
        )

    reference_keys = reference.keys()
    candidate_keys = candidate.keys()
    mismatch_indices = np.flatnonzero(reference_keys != candidate_keys)

    if mismatch_indices.size == 0:
        return

    position = int(mismatch_indices[0])
    reference_key = reference_keys[position]
    candidate_key = candidate_keys[position]

    raise ValueError(
        f"Event indices are not aligned at position {position}: "
        f"reference=({reference_key['file_id']}, "
        f"{reference_key['event_id']}), "
        f"candidate=({candidate_key['file_id']}, "
        f"{candidate_key['event_id']})."
    )
