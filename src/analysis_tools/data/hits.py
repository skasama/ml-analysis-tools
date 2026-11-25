import numpy as np
from numpy.typing import ArrayLike, NDArray

from analysis_tools.data.arrays import as_integer_array


def hit_counts_from_indices(
    event_hits_index: ArrayLike,
    total_hit_count: int,
) -> NDArray[np.int64]:
    """Calculate the number of hits in each event.

    Parameters
    ----------
    event_hits_index
        Starting index of the hits for each event.
    total_hit_count
        Total number of rows in the hit arrays.

    Returns
    -------
    NDArray[np.int64]
        Number of hits in each event.
    """
    event_hits_index = as_integer_array(
        event_hits_index,
        name="event_hits_index",
    )

    if total_hit_count < 0:
        raise ValueError("total_hit_count must be non-negative.")

    if len(event_hits_index) == 0:
        return np.empty(
            0,
            dtype=np.int64,
        )

    if event_hits_index[0] != 0:
        raise ValueError("The first event hit index must be zero.")

    if np.any(np.diff(event_hits_index) < 0):
        raise ValueError("event_hits_index must be monotonically increasing.")

    if event_hits_index[-1] > total_hit_count:
        raise ValueError("event_hits_index exceeds the total hit count.")

    event_boundaries = np.append(
        event_hits_index,
        total_hit_count,
    )

    return np.diff(event_boundaries).astype(np.int64)
