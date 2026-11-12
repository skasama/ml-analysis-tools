import numpy as np


def scalar(x, name="values"):
    x = np.asarray(x)
    if x.ndim == 1:
        return x.astype(float)
    if x.ndim == 2 and x.shape[1] == 1:
        return x[:, 0].astype(float)
    raise ValueError(f"{name} must have shape (N,) or (N,1), got {x.shape}")


def xyz(x, name="values"):
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == 3:
        return x.astype(float)
    if x.ndim == 3 and x.shape[1:] == (1, 3):
        return x[:, 0].astype(float)
    raise ValueError(f"{name} must have shape (N,3) or (N,1,3), got {x.shape}")


def inverse_minmax(x, lo, hi, *, source_min=0.0, source_max=1.0):
    """Map values from an arbitrary normalized interval into ``[lo, hi]``."""
    if np.any(np.asarray(source_max) <= np.asarray(source_min)):
        raise ValueError("source_max must be greater than source_min")
    values = np.asarray(x, dtype=float)
    fraction = (values - source_min) / (source_max - source_min)
    return np.asarray(lo) + fraction * (np.asarray(hi) - np.asarray(lo))
