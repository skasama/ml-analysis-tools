import numpy as np
from numpy.typing import ArrayLike, NDArray


def as_scalar_array(
    values: ArrayLike,
    *,
    name: str = "values",
) -> NDArray:
    """Convert values to a one-dimensional array."""
    array = np.asarray(values)

    if array.ndim == 1:
        return array

    if array.ndim >= 2 and all(size == 1 for size in array.shape[1:]):
        return array.reshape(array.shape[0])

    raise ValueError(
        f"{name} must have shape (N,) or singleton trailing dimensions, "
        f"got {array.shape}."
    )


def as_vector_array(
    values: ArrayLike,
    *,
    size: int = 3,
    name: str = "values",
) -> NDArray:
    """Convert values to an array with shape (N, size)."""
    if size < 1:
        raise ValueError("Vector size must be positive.")

    array = np.asarray(values)

    if array.ndim == 2 and array.shape[1] == size:
        return array

    if array.ndim >= 3 and array.shape[-1] == size:
        middle_shape = array.shape[1:-1]

        if all(dimension == 1 for dimension in middle_shape):
            return array.reshape(array.shape[0], size)

    raise ValueError(
        f"{name} must have shape (N, {size}) or singleton middle dimensions, "
        f"got {array.shape}."
    )


def validate_same_length(
    *arrays: ArrayLike,
    names: tuple[str, ...] | None = None,
) -> int:
    """Validate that all arrays have the same first-axis length."""
    if not arrays:
        raise ValueError("At least one array is required.")

    if names is not None and len(names) != len(arrays):
        raise ValueError("names must have the same length as arrays.")

    lengths = [len(np.asarray(array)) for array in arrays]

    if len(set(lengths)) != 1:
        labels = names or tuple(f"array_{index}" for index in range(len(arrays)))
        details = ", ".join(
            f"{name}={length}" for name, length in zip(labels, lengths, strict=True)
        )
        raise ValueError(f"Array lengths do not match: {details}")

    return lengths[0]
