from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .paths import input_path


@dataclass(frozen=True)
class RunOutputs:
    predictions: np.ndarray
    targets: np.ndarray
    indices: np.ndarray | None


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
