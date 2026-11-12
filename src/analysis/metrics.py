import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

from .geometry import cylindrical_radius, towall, wall


def resolution68(values):
    """Return the 68th percentile of absolute finite values."""
    values = np.asarray(values, dtype=float)
    values = np.abs(values[np.isfinite(values)])
    return float(np.quantile(values, 0.68)) if values.size else np.nan


def correlation(predictions, targets, *, method="pearson"):
    """Return a finite correlation coefficient for two one-dimensional arrays."""
    predictions = np.asarray(predictions, dtype=float).ravel()
    targets = np.asarray(targets, dtype=float).ravel()
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape")
    valid = np.isfinite(predictions) & np.isfinite(targets)
    predictions, targets = predictions[valid], targets[valid]
    if predictions.size < 2:
        return np.nan
    if method == "pearson":
        if np.ptp(predictions) == 0 or np.ptp(targets) == 0:
            return np.nan
        return float(np.corrcoef(predictions, targets)[0, 1])
    if method == "spearman":
        prediction_ranks = np.argsort(np.argsort(predictions))
        target_ranks = np.argsort(np.argsort(targets))
        return correlation(prediction_ranks, target_ranks)
    raise ValueError("method must be 'pearson' or 'spearman'")


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
        "pearson_r": correlation(predictions, targets),
    }


def vertex_metrics(
    predictions,
    targets,
    *,
    directions=None,
    radius=1690.0,
    half_height=1810.0,
):
    """Calculate Cartesian, cylindrical, Wall, and optional ToWall metrics."""
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if (
        predictions.shape != targets.shape
        or predictions.ndim != 2
        or predictions.shape[1] != 3
    ):
        raise ValueError("predictions and targets must both have shape (N, 3)")

    prediction_radius = cylindrical_radius(predictions)
    target_radius = cylindrical_radius(targets)
    prediction_wall = wall(predictions, radius=radius, half_height=half_height)
    target_wall = wall(targets, radius=radius, half_height=half_height)
    values = {
        "x": regression_metrics(predictions[:, 0], targets[:, 0]),
        "y": regression_metrics(predictions[:, 1], targets[:, 1]),
        "z": regression_metrics(predictions[:, 2], targets[:, 2]),
        "r": regression_metrics(prediction_radius, target_radius),
        "wall": regression_metrics(prediction_wall, target_wall),
        "error_3d": regression_metrics(
            np.linalg.norm(predictions - targets, axis=1),
            np.zeros(len(predictions)),
        ),
    }
    if directions is not None:
        directions = np.asarray(directions, dtype=float)
        if directions.shape != predictions.shape:
            raise ValueError("directions must have shape (N, 3)")
        values["towall"] = regression_metrics(
            towall(predictions, directions, radius=radius, half_height=half_height),
            towall(targets, directions, radius=radius, half_height=half_height),
        )
    return values


def classification_metrics(predictions, targets):
    """Calculate accuracy and one-vs-rest class-wise classification metrics."""
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
    true_positive = np.diag(confusion).astype(float)
    false_negative = confusion.sum(axis=1) - true_positive
    false_positive = confusion.sum(axis=0) - true_positive
    true_negative = confusion.sum() - true_positive - false_negative - false_positive

    def safe_divide(numerator, denominator):
        return np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan, dtype=float),
            where=denominator != 0,
        )

    return {
        "n_events": int(len(target_labels)),
        "accuracy": float(accuracy_score(target_labels, predicted_labels)),
        "labels": labels,
        "confusion_matrix": confusion,
        "efficiency": safe_divide(true_positive, true_positive + false_negative),
        "purity": safe_divide(true_positive, true_positive + false_positive),
        "mis_pid": safe_divide(false_positive, false_positive + true_negative),
    }


def compare_metrics(results, metric_function):
    """Calculate the same metrics for multiple MethodResult objects."""
    return {
        result.method: metric_function(result.predictions, result.targets)
        for result in results
    }
