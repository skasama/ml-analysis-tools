from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray


def load_npy(
    path: str | Path,
    *,
    allow_pickle: bool = False,
) -> NDArray:
    """

    Load a NumPy array from a .npy file.

    Parameters
    ----------
    path : str | Path
        Path to the .npy file.
    allow_pickle : bool, optional
        Whether to allow loading pickled object arrays. Default is False.

    Returns
    -------
    NDArray
        The loaded NumPy array.
    """
    path = _validate_input_path(path, suffixes={".npy"})
    array = np.load(path, allow_pickle=allow_pickle)
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Expected a NumPy array in {path}.")

    return array


def save_npy(
    path: str | Path,
    array: ArrayLike,
    *,
    create_parents: bool = True,
) -> Path:
    """

    Save an array to a .npy file.

    Parameters
    ----------
    path : str | Path
        Path to the .npy file.
    array : ArrayLike
        The array to save.
    create_parents : bool, optional
        Whether to create parent directories if they don't exist. Default is True.

    Returns
    -------
    Path
        The path to the saved .npy file.
    Raises
    ------
    ValueError
        If the provided path does not have a .npy extension.
    FileNotFoundError
        If the parent directory of the provided path does not exist and create_parents
        is False.
    """
    path = _prepare_output_path(path, suffix=".npy", create_parents=create_parents)

    np.save(path, np.asanyarray(array))
    return path


def load_npz(
    path: str | Path,
    *,
    keys: Sequence[str] | None = None,
    allow_pickle: bool = False,
) -> dict[str, NDArray]:
    """
    Load arrays from a .npz archive.

    If keys are provided, only those arrays are loaded.

    Parameters
    ----------
    path : str | Path
        Path to the .npz file.
    keys : Sequence[str] | None, optional
        Specific keys of arrays to load. If None, all arrays are loaded.
        Default is None.
    allow_pickle : bool, optional
        Whether to allow loading pickled object arrays. Default is False.

    Returns
    -------
    dict[str, NDArray]
        A dictionary mapping keys to loaded NumPy arrays.
    """
    path = _validate_input_path(path, suffixes={".npz"})

    with np.load(path, allow_pickle=allow_pickle) as archive:
        available_keys = set(archive.files)

        if keys is None:
            selected_keys = archive.files
        else:
            missing_keys = set(keys) - available_keys
            if missing_keys:
                missing = ", ".join(missing_keys)
                available = ", ".join(available_keys)

                raise KeyError(
                    f"Missing keys in {path}: {missing}. Available keys: {available}."
                )

            selected_keys = keys
        return {key: np.asarray(archive[key]) for key in selected_keys}


def save_npz(
    path: str | Path,
    arrays: Mapping[str, ArrayLike],
    *,
    compressed: bool = False,
    create_parents: bool = True,
) -> Path:
    """

    Save named arrays to a .npz archive.

    Parameters
    ----------
    path : str | Path
        Path to the .npz file.
    arrays : Mapping[str, ArrayLike]
        A mapping of names to arrays to save.
    compressed : bool, optional
        Whether to use compressed .npz format. Default is False.
    create_parents : bool, optional
        Whether to create parent directories if they don't exist. Default is True.

    Returns
    -------
    Path
        The path to the saved .npz file.

    Raises
    ------
    ValueError
        If the provided path does not have a .npz extension.
    FileNotFoundError
        If the parent directory of the provided path does not exist
        and create_parents is False.
    """
    if not arrays:
        raise ValueError("At least one array is required.")

    path = _prepare_output_path(
        path,
        suffix=".npz",
        create_parents=create_parents,
    )

    converted = {key: np.asarray(array) for key, array in arrays.items()}

    if compressed:
        np.savez_compressed(path, **converted)
    else:
        np.savez(path, **converted)

    return path


def _validate_input_path(
    path: str | Path,
    *,
    suffixes: set[str],
) -> Path:
    path = Path(path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() not in suffixes:
        expected = ",".join(sorted(suffixes))
        raise ValueError(
            f"Unsupported file extension for {path}.Expected one of: {expected}."
        )

    return path


def _prepare_output_path(
    path: str | Path,
    *,
    suffix: str,
    create_parents: bool,
) -> Path:
    path = Path(path).expanduser()

    if path.suffix.lower() != suffix:
        raise ValueError(f"Output path must use the {suffix} extension: {path}")

    if create_parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    elif not path.parent.exists():
        raise FileNotFoundError(path.parent)

    return path
