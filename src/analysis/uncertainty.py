from collections.abc import Callable

import numpy as np


def profile_uncertainty(
    x_values,
    residuals,
    *,
    n_bins,
    n_samples=200,
    confidence=0.68,
    seed=0,
):
    """Estimate bootstrap intervals for binned bias and 68% resolution."""
    x_values = np.asarray(x_values, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    if x_values.shape != residuals.shape:
        raise ValueError("x_values and residuals must have the same shape")
    if n_bins <= 0 or n_samples <= 0:
        raise ValueError("n_bins and n_samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    finite = np.isfinite(x_values) & np.isfinite(residuals)
    x_values = x_values[finite]
    residuals = residuals[finite]
    if x_values.size == 0:
        raise ValueError("No finite values are available for profile uncertainty.")

    edges = np.linspace(np.min(x_values), np.max(x_values), n_bins + 1)
    if edges[-1] <= edges[0]:
        edges[-1] = edges[0] + np.finfo(float).eps

    bias_intervals = np.full((2, n_bins), np.nan)
    resolution_intervals = np.full((2, n_bins), np.nan)
    lower_quantile = (1.0 - confidence) / 2.0
    upper_quantile = 1.0 - lower_quantile
    rng = np.random.default_rng(seed)

    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        in_bin = (x_values >= low) & (
            (x_values <= high) if index == n_bins - 1 else (x_values < high)
        )
        values = residuals[in_bin]
        if values.size == 0:
            continue
        selected = rng.integers(0, values.size, size=(n_samples, values.size))
        samples = values[selected]
        biases = np.mean(samples, axis=1)
        resolutions = np.quantile(np.abs(samples), 0.68, axis=1)
        bias_intervals[:, index] = np.quantile(biases, [lower_quantile, upper_quantile])
        resolution_intervals[:, index] = np.quantile(
            resolutions, [lower_quantile, upper_quantile]
        )

    return {
        "bias": bias_intervals,
        "resolution": resolution_intervals,
    }


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
