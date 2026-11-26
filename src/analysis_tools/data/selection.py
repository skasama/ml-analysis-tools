from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from analysis_tools.data.arrays import validate_same_length


def range_mask(
    values: ArrayLike,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    inclusive: bool = True,
) -> NDArray[np.bool_]:
    """

    Create a mask selecting values within a range.

    Parameters
    ----------
    values : ArrayLike
        Input array of values to be masked.
    minimum : float | None, optional
        Minimum value for the range. If None, no lower bound is applied.
    maximum : float | None, optional
        Maximum value for the range. If None, no upper bound is applied.
    inclusive : bool, optional
        If True, the range is inclusive of the minimum and maximum values.
        If False, the range is exclusive. Default is True.

    Returns
    -------
    NDArray[np.bool_]
        A boolean array with the same length as the input arrays, where True indicates
        that the corresponding value is within the specified range.
        Raises ValueError if the input values are not one-dimensional.
    """
    values = np.asarray(values)

    if values.ndim != 1:
        raise ValueError(f"values must be one-dimensional, got shape {values.shape}.")

    mask = np.ones(len(values), dtype=bool)

    if minimum is not None:
        mask &= values >= minimum if inclusive else values > minimum

    if maximum is not None:
        mask &= values <= maximum if inclusive else values < maximum

    return mask


def combine_masks(
    *masks: ArrayLike,
) -> NDArray[np.bool_]:
    """Combine selection masks using logical AND."""
    if not masks:
        raise ValueError("At least one mask is required.")

    arrays = [np.asarray(mask, dtype=bool) for mask in masks]
    validate_same_length(*arrays)

    for array in arrays:
        if array.ndim != 1:
            raise ValueError(f"Masks must be one-dimensional, got shape {array.shape}.")

    return np.logical_and.reduce(arrays)


def apply_mask(
    mask: ArrayLike,
    *arrays: ArrayLike,
) -> tuple[NDArray, ...]:
    """Apply one selection mask to multiple arrays."""
    mask = np.asarray(mask, dtype=bool)

    if mask.ndim != 1:
        raise ValueError(f"mask must be one-dimensional, got shape {mask.shape}.")

    converted = [np.asarray(array) for array in arrays]
    validate_same_length(mask, *converted)

    return tuple(array[mask] for array in converted)


def apply_mask_to_mapping(
    mask: ArrayLike,
    arrays: Mapping[str, ArrayLike],
) -> dict[str, NDArray]:
    """Apply one selection mask to a mapping of named arrays."""
    keys = list(arrays)
    selected = apply_mask(mask, *(arrays[key] for key in keys))

    return dict(zip(keys, selected, strict=True))


@dataclass(frozen=True)
class CutFlowEntry:
    """

    Event counts before and after one cut.

    Parameters
    ----------
    name : str
        Name of the cut.
    before : int
        Number of events before the cut.
    after : int
        Number of events after the cut.
    """

    name: str
    before: int
    after: int

    @property
    def removed(self) -> int:
        """Return the number of removed events."""
        return self.before - self.after


class EventSelection:
    """

    Build an event selection from Boolean masks.

    Parameters
    ----------
    event_count : int
        The total number of events in the selection.
    """

    def __init__(
        self,
        event_count: int,
    ) -> None:
        if event_count < 0:
            raise ValueError("event_count must be non-negative.")

        self._mask = np.ones(
            event_count,
            dtype=bool,
        )

        self._cut_flow: list[CutFlowEntry] = []

    @property
    def mask(self) -> NDArray[np.bool_]:
        """Return a copy of the current selection mask."""
        return self._mask.copy()

    @property
    def indices(self) -> NDArray[np.int64]:
        """Return the selected event indices."""
        return np.flatnonzero(self._mask).astype(
            np.int64,
            copy=False,
        )

    @property
    def event_count(self) -> int:
        """Return the number of selected events."""
        return int(np.count_nonzero(self._mask))

    @property
    def cut_flow(
        self,
    ) -> tuple[CutFlowEntry, ...]:
        """Return the recorded cut flow."""
        return tuple(self._cut_flow)

    def add(
        self,
        name: str,
        condition: ArrayLike,
    ) -> "EventSelection":
        """

        Apply one Boolean condition.

        Parameters
        ----------
        name : str
            Name of the cut.
        condition : ArrayLike
            The Boolean condition to apply.
        """
        condition = np.asarray(condition)

        if condition.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional.")

        if condition.dtype != np.bool_:
            raise ValueError(f"{name} must be a Boolean array.")

        if len(condition) != len(self._mask):
            raise ValueError(
                f"{name} has {len(condition)} events, "
                f"but the selection has "
                f"{len(self._mask)}."
            )

        before = self.event_count

        self._mask &= condition

        after = self.event_count

        self._cut_flow.append(
            CutFlowEntry(
                name=name,
                before=before,
                after=after,
            )
        )

        return self

    def print_cut_flow(self) -> None:
        """Print the recorded cut flow."""
        print("Cut flow:")
        print(f"  {'Initial events':30s} {len(self._mask):10,d}")

        for entry in self._cut_flow:
            print(
                f"  {entry.name:30s} "
                f"{entry.before:10,d} -> "
                f"{entry.after:10,d}  "
                f"removed: {entry.removed:10,d}"
            )
