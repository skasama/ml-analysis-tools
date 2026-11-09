import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error


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
