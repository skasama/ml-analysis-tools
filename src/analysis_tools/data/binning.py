from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class BinnedSelection:
    """

    A binned selection of events.

    Attributes
    ----------
    edges : NDArray[np.float64]
        The bin edges.
    centers : NDArray[np.float64]
        The bin centers.
    masks : tuple[NDArray[np.bool_], ...]
        A tuple of boolean masks, one for each bin.
    counts : NDArray[np.int_]
        The number of events in each bin.
    """

    edges: NDArray[np.float64]
    centers: NDArray[np.float64]
    masks: tuple[NDArray[np.bool_], ...]
    counts: NDArray[np.int64]

    @property
    def number_of_bins(self) -> int:
        """Return the number of bins."""
        return len(self.centers)

    @property
    def event_count(self) -> int:
        """Return the number of input events."""
        if not self.masks:
            return 0

        return len(self.masks[0])

    def mask(
        self,
        index: int,
    ) -> NDArray[np.bool_]:
        """Return a copy of one bin mask."""
        return self.masks[index].copy()

    def indices(
        self,
        index: int,
    ) -> NDArray[np.int64]:
        """Return event indices in one bin."""
        return np.flatnonzero(self.masks[index]).astype(
            np.int64,
            copy=False,
        )

    def valid_bins(
        self,
        *,
        minimum_events: int = 1,
    ) -> NDArray[np.bool_]:
        """Return bins satisfying an event-count requirement."""
        if minimum_events < 1:
            raise ValueError("minimum_events must be at least 1.")

        return self.counts >= minimum_events


def make_binned_selection(
    values: ArrayLike,
    edges: ArrayLike,
    *,
    include_right_edge: bool = True,
) -> BinnedSelection:
    """

    Create a binned selection of events.

    Parameters
    ----------
    values : ArrayLike
        The values to bin.
    edges : ArrayLike
        The bin edges.
    include_right_edge : bool, optional
        Whether to include the right edge of the last bin. Default is True.

    Returns
    -------
    BinnedSelection
        A binned selection of events.
    """
    values = np.asarray(
        values,
        dtype=float,
    )

    edges = np.asarray(
        edges,
        dtype=float,
    )

    if values.ndim != 1:
        raise ValueError("values must be one-dimensional.")

    if edges.ndim != 1:
        raise ValueError("edges must be one-dimensional.")

    if len(edges) < 2:
        raise ValueError("At least two bin edges are required.")

    if not np.all(np.isfinite(edges)):
        raise ValueError("edges must contain only finite values.")

    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("edges must be strictly increasing.")

    centers = 0.5 * (edges[:-1] + edges[1:])

    finite = np.isfinite(values)

    masks: list[NDArray[np.bool_]] = []
    counts = np.zeros(
        len(centers),
        dtype=np.int64,
    )

    for index in range(len(centers)):
        lower = edges[index]
        upper = edges[index + 1]

        if include_right_edge and index == len(centers) - 1:
            mask = finite & (values >= lower) & (values <= upper)
        else:
            mask = finite & (values >= lower) & (values < upper)

        masks.append(mask)

        counts[index] = np.count_nonzero(mask)

    return BinnedSelection(
        edges=edges,
        centers=centers,
        masks=tuple(masks),
        counts=counts,
    )
