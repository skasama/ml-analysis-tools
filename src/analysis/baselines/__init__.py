"""Small non-ML baselines for method comparisons."""

import numpy as np


def constant_prediction(targets, value=None):
    """Return a constant prediction with the same shape as targets."""
    targets = np.asarray(targets)
    if value is None:
        value = np.mean(targets, axis=0)
    return np.broadcast_to(value, targets.shape).copy()


__all__ = ["constant_prediction"]
