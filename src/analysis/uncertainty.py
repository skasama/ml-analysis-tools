from collections.abc import Callable

import numpy as np


def bootstrap_metric(
    predictions,
    targets,
    metric: Callable,
    *,
    n_samples=1000,
    seed=0,
):
    """Estimate a metric and percentile confidence interval by event bootstrap."""
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    rng = np.random.default_rng(seed)
    estimate = metric(predictions, targets)
    samples = np.empty(n_samples, dtype=float)
    for index in range(n_samples):
        selected = rng.integers(0, len(targets), len(targets))
        samples[index] = metric(predictions[selected], targets[selected])
    return {
        "estimate": float(estimate),
        "lower": float(np.quantile(samples, 0.025)),
        "upper": float(np.quantile(samples, 0.975)),
        "samples": samples,
    }
