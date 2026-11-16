from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def load_csv(
    path: str | Path,
    *,
    names: bool = True,
    delimiter: str = ",",
    encoding: str = "utf-8",
) -> NDArray:
    """Load tabular data from a CSV file."""
    path = _validate_csv_path(path)

    array = np.genfromtxt(
        path,
        delimiter=delimiter,
        names=True if names else None,
        encoding=encoding,
    )

    return np.asarray(array)


def save_csv(
    path: str | Path,
    array: NDArray,
    *,
    column_names: Sequence[str] | None = None,
    delimiter: str = ",",
    fmt: str | Sequence[str] = "%.18e",
    create_parents: bool = True,
) -> Path:
    """Save a one- or two-dimensional NumPy array to a CSV file."""
    path = _prepare_output_path(
        path,
        create_parents=create_parents,
    )
    array = np.asarray(array)

    if array.ndim not in {1, 2}:
        raise ValueError(
            f"CSV output requires a one- or two-dimensional array, "
            f"got shape {array.shape}."
        )

    column_count = 1 if array.ndim == 1 else array.shape[1]

    if column_names is not None and len(column_names) != column_count:
        raise ValueError(
            f"Expected {column_count} column names, got {len(column_names)}."
        )

    header = delimiter.join(column_names) if column_names else ""

    np.savetxt(
        path,
        array,
        delimiter=delimiter,
        fmt=fmt,
        header=header,
        comments="",
    )

    return path


def _validate_csv_path(path: str | Path) -> Path:
    path = Path(path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(path)

    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file: {path}")

    return path


def _prepare_output_path(
    path: str | Path,
    *,
    create_parents: bool,
) -> Path:
    path = Path(path).expanduser()

    if path.suffix.lower() != ".csv":
        raise ValueError(f"Output path must use the .csv extension: {path}")

    if create_parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    elif not path.parent.exists():
        raise FileNotFoundError(path.parent)

    return path
