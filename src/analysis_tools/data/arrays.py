import numpy as np
from numpy.typing import ArrayLike, NDArray


def as_scalar_array(
    values: ArrayLike,
    *,
    name: str = "values",
) -> NDArray:
    """

    Convert values to a one-dimensional array.

    Parameters
    ----------
    values : ArrayLike
        Input values to be converted to a one-dimensional array.
    name : str, optional
        Name of the input values for error messages. Default is "values".

    Returns
    -------
    NDArray
        A one-dimensional array containing the input values.
    """
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
    """

    Convert values to an array with shape (N, size).

    Parameters
    ----------
    values : ArrayLike
        Input values to be converted to an array with shape (N, size).
    size : int, optional
        The desired size of the second dimension. Default is 3.
    name : str, optional
        Name of the input values for error messages. Default is "values".

    Returns
    -------
    NDArray
        An array with shape (N, size).
    """
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
    """

    Validate that all arrays have the same first-axis length.

    Parameters
    ----------
    *arrays : ArrayLike
        Input arrays to be validated for the same first-axis length.
    names : tuple[str, ...], optional
        Names of the input arrays for error messages. Default is None.

    Returns
    -------
    int
        The common length of the first axis of all arrays.

    Raises
    ------
    ValueError
        If the arrays do not have the same first-axis length.
    """
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


def as_integer_array(
    values: ArrayLike,
    *,
    name: str = "values",
) -> NDArray[np.int64]:
    """

    Convert integer-valued data to a one-dimensional integer array.

    Parameters
    ----------
    values : ArrayLike
        Input values to be converted to a one-dimensional integer array.
    name : str, optional
        Name of the input values for error messages. Default is "values".
    Returns
    -------
    NDArray[np.int64]
        A one-dimensional integer array.
    """
    array = as_scalar_array(values, name=name)

    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must contain integer values.")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite integer values.")

    integer_array = array.astype(np.int64)

    if not np.array_equal(array, integer_array):
        raise ValueError(f"{name} must contain integer values.")

    return integer_array
