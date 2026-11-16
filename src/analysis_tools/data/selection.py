from collections.abc import Mapping

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
    """Create a mask selecting values within a range."""
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
