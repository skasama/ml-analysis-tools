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


def inverse_minmax(x, lo, hi):
    return np.asarray(lo) + np.asarray(x) * (np.asarray(hi) - np.asarray(lo))
