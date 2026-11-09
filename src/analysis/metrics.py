import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error


def resolution68(values):
    """Return the 68th percentile of absolute finite values."""
    values = np.asarray(values, dtype=float)
    values = np.abs(values[np.isfinite(values)])
    return float(np.quantile(values, 0.68)) if values.size else np.nan


def residual_profile(x_values, residuals, *, n_bins):
    """Calculate binned bias, 68% resolution, and event counts."""
    x_values = np.asarray(x_values, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(residuals)
    x_values, residuals = x_values[finite], residuals[finite]
    if x_values.size == 0:
        raise ValueError("No finite values are available for profile calculation.")

    edges = np.linspace(np.min(x_values), np.max(x_values), n_bins + 1)
    if edges[-1] <= edges[0]:
        edges[-1] = edges[0] + np.finfo(float).eps
    centres = 0.5 * (edges[:-1] + edges[1:])
    biases = np.full(n_bins, np.nan)
    resolutions = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        in_bin = (x_values >= low) & (
            (x_values <= high) if index == n_bins - 1 else (x_values < high)
        )
        values = residuals[in_bin]
        counts[index] = len(values)
        if values.size:
            biases[index] = np.mean(values)
            resolutions[index] = resolution68(values)
    return centres, biases, resolutions, counts


def regression_metrics(predictions, targets):
    """Calculate method-independent scalar regression metrics."""
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape")
    residuals = predictions - targets
    absolute = np.abs(residuals)
    return {
        "n_events": int(len(predictions)),
        "bias": float(np.mean(residuals)),
        "mae": float(mean_absolute_error(targets, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(targets, predictions))),
        "resolution_68": float(np.quantile(absolute, 0.68)),
        "resolution_90": float(np.quantile(absolute, 0.90)),
        "outlier_rate_3sigma": float(np.mean(absolute > 3.0 * np.std(residuals))),
    }


def classification_metrics(predictions, targets):
    """Calculate accuracy and class-wise classification counts."""
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    predicted_labels = (
        np.argmax(predictions, axis=1)
        if predictions.ndim == 2
        else predictions.astype(int)
    )
    target_labels = (
        np.argmax(targets, axis=1) if targets.ndim == 2 else targets.astype(int)
    )
    if predicted_labels.shape != target_labels.shape:
        raise ValueError("predictions and targets must describe the same events")
    labels = np.unique(np.concatenate((target_labels, predicted_labels)))
    confusion = np.zeros((len(labels), len(labels)), dtype=int)
    for target, predicted in zip(target_labels, predicted_labels, strict=True):
        target_index = np.flatnonzero(labels == target)[0]
        predicted_index = np.flatnonzero(labels == predicted)[0]
        confusion[target_index, predicted_index] += 1
    return {
        "n_events": int(len(target_labels)),
        "accuracy": float(accuracy_score(target_labels, predicted_labels)),
        "labels": labels,
        "confusion_matrix": confusion,
    }


def compare_metrics(results, metric_function):
    """Calculate the same metrics for multiple MethodResult objects."""
    return {
        result.method: metric_function(result.predictions, result.targets)
        for result in results
    }
