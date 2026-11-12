from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .paths import input_path


@dataclass(frozen=True)
class RunOutputs:
    predictions: np.ndarray
    targets: np.ndarray
    indices: np.ndarray | None


@dataclass(frozen=True)
class InputVariable:
    """A selectable array stored in an analysis input file."""

    path: Path
    variable: str | None
    shape: tuple[int, ...]
    dtype: str


def _iter_hdf5_datasets(group, prefix="") -> Iterator[tuple[str, h5py.Dataset]]:
    for name, value in group.items():
        dataset_name = f"{prefix}/{name}" if prefix else name
        if isinstance(value, h5py.Dataset):
            yield dataset_name, value
        elif isinstance(value, h5py.Group):
            yield from _iter_hdf5_datasets(value, dataset_name)


def list_input_variables(
    run_dir,
    patterns=("*.npy", "*.npz", "*.csv", "*.h5", "*.hdf5", "*.hy"),
):
    """List selectable arrays below ``run_dir`` without loading their contents."""
    run_dir = Path(run_dir)
    variables = []
    for path in sorted(
        candidate
        for pattern in patterns
        for candidate in run_dir.glob(f"**/{pattern}")
        if candidate.is_file()
        and not any(
            part.startswith(".") for part in candidate.relative_to(run_dir).parts
        )
    ):
        suffix = path.suffix.lower()
        if suffix == ".npy":
            try:
                array = np.load(path, allow_pickle=False)
            except ValueError:
                continue
            variables.append(InputVariable(path, None, array.shape, str(array.dtype)))
        elif suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                variables.extend(
                    InputVariable(
                        path,
                        name,
                        archive[name].shape,
                        str(archive[name].dtype),
                    )
                    for name in archive.files
                )
        elif suffix in {".h5", ".hdf5", ".hy"}:
            with h5py.File(path, "r") as handle:
                variables.extend(
                    InputVariable(path, name, tuple(dataset.shape), str(dataset.dtype))
                    for name, dataset in _iter_hdf5_datasets(handle)
                )
        else:
            array = np.genfromtxt(path, delimiter=",", names=True)
            shape = array.shape if array.dtype.names else tuple(array.shape)
            variables.append(InputVariable(path, None, shape, str(array.dtype)))
    return tuple(variables)


def load_variable(path, variable=None):
    """Load one array from NPY, NPZ, CSV, or HDF5 input."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        if variable is not None:
            raise ValueError("NPY files do not contain selectable variables")
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            name = variable or (archive.files[0] if len(archive.files) == 1 else None)
            if name is None:
                raise ValueError(f"Variable is required for multi-array file: {path}")
            if name not in archive:
                raise KeyError(f"Variable {name!r} was not found in {path}")
            return archive[name]
    if suffix in {".h5", ".hdf5", ".hy"}:
        if not variable:
            raise ValueError(f"Variable is required for HDF5 file: {path}")
        with h5py.File(path, "r") as handle:
            if variable not in handle:
                raise KeyError(f"Variable {variable!r} was not found in {path}")
            return np.asarray(handle[variable])
    if suffix == ".csv":
        return np.genfromtxt(
            path,
            delimiter=",",
            names=variable is not None,
            dtype=float,
        )
    raise ValueError(f"Unsupported input format: {path.suffix}")


def load_outputs(
    run_dir,
    predictions="outputs/preds.npy",
    targets="outputs/targets.npy",
    indices="outputs/indices.npy",
    require_indices=False,
):
    pp = input_path(run_dir, predictions, "Predictions file")
    tp = input_path(run_dir, targets, "Targets file")
    p = np.load(pp, allow_pickle=False)
    t = np.load(tp, allow_pickle=False)
    if len(p) != len(t):
        raise ValueError(f"Event count mismatch: {len(p)} != {len(t)}")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(t)):
        raise ValueError("Predictions or targets contain non-finite values")
    idx = None
    if indices:
        candidate = Path(indices)
        candidate = candidate if candidate.is_absolute() else Path(run_dir) / candidate
        if candidate.is_file():
            idx = np.load(candidate, allow_pickle=False)
        elif require_indices:
            raise FileNotFoundError(
                f"Indices file was not found: {candidate.resolve()}"
            )
    if idx is not None and len(idx) != len(t):
        raise ValueError("Index count mismatch")
    return RunOutputs(p, t, idx)


def assert_aligned(*items, require_indices=False):
    if len({len(x.targets) for x in items}) != 1:
        raise ValueError("Runs have different event counts")
    if require_indices and any(x.indices is None for x in items):
        raise ValueError("indices.npy is required for every run")
    known = [x.indices for x in items if x.indices is not None]
    if len(known) > 1 and any(not np.array_equal(known[0], x) for x in known[1:]):
        raise ValueError("indices.npy differs between runs")
