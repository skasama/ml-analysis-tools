from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from analysis_tools.data.arrays import (
    as_integer_array,
    as_scalar_array,
    validate_same_length,
)


def predicted_labels(
    scores: ArrayLike,
) -> NDArray[np.int64]:
    """
    Determine the predicted class labels based on the highest scores.
    Return the class with the highest score for each event.

    Parameters
    ----------
    scores : ArrayLike
        A 2D array of shape (N, C) where N is the number of events and C is
        the number of classes. Each row contains the scores for each class.

    Returns
    -------
    NDArray[np.int64]
        A 1D array of shape (N,) containing the predicted class labels for each event.
    """
    scores = np.asarray(scores)

    if scores.ndim != 2:
        raise ValueError(f"Scores must have shape (N, C), got {scores.shape}.")
    if scores.shape[1] < 2:
        raise ValueError("scores must contain at lease two classes.")

    return np.argmax(scores, axis=1).astype(np.int64)


def confusion_matrix(
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    class_labels: Sequence[int] | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """

    Calculate a confusion matrix with truth on rows.

    Parameters
    ----------
    predictions : ArrayLike
        Predicted class labels.
    targets : ArrayLike
        True class labels.
    class_labels : Sequence[int], optional
        The labels of the classes. If None, the class labels will be inferred
        from the maximum class label in predictions and targets.

    Returns
    -------
    NDArray[np.int64]
        A 2D array of shape (num_classes, num_classes) representing
        the confusion matrix.
    """
    predictions = as_integer_array(
        predictions,
        name="predictions",
    ).astype(np.int64)
    targets = as_integer_array(
        targets,
        name="targets",
    ).astype(np.int64)

    validate_same_length(
        predictions,
        targets,
        names=("predictions", "targets"),
    )

    if class_labels is None:
        labels = np.unique(np.concatenate((targets, predictions)))
    else:
        labels = _as_integer_labels(
            class_labels,
            name="class_labels",
        )

        if len(np.unique(labels)) != len(labels):
            raise ValueError("class_labels must be unique.")

    label_to_index = {int(label): index for index, label in enumerate(labels)}

    unknown = set(predictions) | set(targets)
    unknown -= set(labels)

    if unknown:
        raise ValueError(f"Labels are missing from class_labels: {sorted(unknown)}")

    prediction_indices = np.fromiter(
        (label_to_index[int(label)] for label in predictions),
        dtype=np.int64,
        count=len(predictions),
    )
    target_indices = np.fromiter(
        (label_to_index[int(label)] for label in targets),
        dtype=np.int64,
        count=len(targets),
    )

    matrix = np.zeros(
        (len(labels), len(labels)),
        dtype=np.int64,
    )
    np.add.at(
        matrix,
        (target_indices, prediction_indices),
        1,
    )

    return matrix, labels


def classification_efficiency(
    matrix: ArrayLike,
) -> NDArray[np.float64]:
    """

    Calculate per-class efficiency from a confusion matrix.

    Parameters
    ----------
    matrix : ArrayLike
        A confusion matrix with truth on rows.

    Returns
    -------
    NDArray[np.float64]
        A 1D array of shape (num_classes,) containing the efficiency for each class.
    """
    matrix = _validate_confusion_matrix(matrix)
    totals = matrix.sum(axis=1)

    return _safe_diagonal_ratio(matrix, totals)


def classification_purity(
    matrix: ArrayLike,
) -> NDArray[np.float64]:
    """

    Calculate per-class purity from a confusion matrix.

    Parameters
    ----------
    matrix : ArrayLike
        A confusion matrix with truth on rows.

    Returns
    -------
    NDArray[np.float64]
        A 1D array of shape (num_classes,) containing the purity for each class.
    """
    matrix = _validate_confusion_matrix(matrix)
    totals = matrix.sum(axis=0)

    return _safe_diagonal_ratio(matrix, totals)


def softmax(
    scores: ArrayLike,
) -> NDArray[np.float64]:
    """Calculate numerically stable softmax probabilities."""
    scores = np.asarray(scores, dtype=float)

    if scores.ndim != 2:
        raise ValueError(f"scores must have shape (N, C), got {scores.shape}.")

    if scores.shape[1] < 2:
        raise ValueError("scores must contain at least two classes.")

    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must contain only finite values.")

    shifted_scores = scores - np.max(
        scores,
        axis=1,
        keepdims=True,
    )
    exponentials = np.exp(shifted_scores)

    return exponentials / np.sum(
        exponentials,
        axis=1,
        keepdims=True,
    )


def classification_accuracy(
    predictions: ArrayLike,
    targets: ArrayLike,
) -> float:
    """Calculate the fraction of correctly classified events."""
    predictions = as_integer_array(
        predictions,
        name="predictions",
    )
    targets = as_integer_array(
        targets,
        name="targets",
    )

    validate_same_length(
        predictions,
        targets,
        names=("predictions", "targets"),
    )

    if len(targets) == 0:
        raise ValueError("At least one event is required.")

    return float(np.mean(predictions == targets))


def class_metrics(
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    class_labels: ArrayLike | None = None,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Calculate per-class efficiency and purity.

    Parameters
    ----------
    predictions : ArrayLike
        The predicted class labels.
    targets : ArrayLike
        The true class labels.
    class_labels : ArrayLike | None, optional
        The labels of the classes, by default None.

    Returns
    -------
    tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]
        A tuple of the class labels, efficiency, and purity.
    """
    matrix, labels = confusion_matrix(
        predictions,
        targets,
        class_labels=class_labels,
    )

    efficiency = classification_efficiency(matrix)
    purity = classification_purity(matrix)

    return labels, efficiency, purity


def roc_curve(
    scores: ArrayLike,
    targets: ArrayLike,
    *,
    positive_label: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Calculate a one-vs-rest ROC curve.

    Parameters
    ----------
    scores : ArrayLike
        The predicted scores for the positive class.
    targets : ArrayLike
        The true class labels.
    positive_label : int
        The label of the positive class.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        A tuple of false-positive rates, true-positive rates, and thresholds.
    """
    scores = as_scalar_array(
        scores,
        name="scores",
    ).astype(float)
    targets = as_integer_array(
        targets,
        name="targets",
    )

    validate_same_length(
        scores,
        targets,
        names=("scores", "targets"),
    )

    finite = np.isfinite(scores)
    scores = scores[finite]
    targets = targets[finite]

    positive = targets == positive_label
    negative = ~positive

    positive_count = int(np.count_nonzero(positive))
    negative_count = int(np.count_nonzero(negative))

    if positive_count == 0:
        raise ValueError(f"No events have positive label {positive_label}.")

    if negative_count == 0:
        raise ValueError("At least one negative event is required.")

    order = np.argsort(scores, kind="stable")[::-1]
    sorted_scores = scores[order]
    sorted_positive = positive[order]

    true_positives = np.cumsum(sorted_positive)
    false_positives = np.cumsum(~sorted_positive)

    distinct_indices = np.where(np.diff(sorted_scores) != 0.0)[0]
    threshold_indices = np.concatenate(
        (
            distinct_indices,
            np.array([len(sorted_scores) - 1]),
        )
    )

    true_positive_rates = true_positives[threshold_indices] / positive_count
    false_positive_rates = false_positives[threshold_indices] / negative_count
    thresholds = sorted_scores[threshold_indices]

    true_positive_rates = np.concatenate(([0.0], true_positive_rates))
    false_positive_rates = np.concatenate(([0.0], false_positive_rates))
    thresholds = np.concatenate(([np.inf], thresholds))

    return (
        false_positive_rates.astype(float),
        true_positive_rates.astype(float),
        thresholds.astype(float),
    )


def auc(
    x: ArrayLike,
    y: ArrayLike,
) -> float:
    """Calculate area under a curve using trapezoidal integration."""
    x = as_scalar_array(x, name="x").astype(float)
    y = as_scalar_array(y, name="y").astype(float)

    validate_same_length(
        x,
        y,
        names=("x", "y"),
    )

    if len(x) < 2:
        raise ValueError("At least two curve points are required.")

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must contain only finite values.")

    if np.any(np.diff(x) < 0.0):
        raise ValueError("x must be monotonically non-decreasing.")

    return float(np.trapezoid(y, x))


def roc_auc(
    scores: ArrayLike,
    targets: ArrayLike,
    *,
    positive_label: int,
) -> float:
    """

    Calculate one-vs-rest ROC AUC.

    Parameters
    ----------
    scores : ArrayLike
        The predicted scores for the positive class.
    targets : ArrayLike
        The true class labels.
    positive_label : int
        The label of the positive class.

    Returns
    -------
    float
        The one-vs-rest ROC AUC.
    """
    false_positive_rates, true_positive_rates, _ = roc_curve(
        scores,
        targets,
        positive_label=positive_label,
    )

    return auc(
        false_positive_rates,
        true_positive_rates,
    )


def threshold_metrics(
    scores: ArrayLike,
    targets: ArrayLike,
    thresholds: ArrayLike,
    *,
    positive_label: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Calculate efficiency and purity versus score threshold.

    An event is selected when score >= threshold.

    Returns thresholds, efficiency, and purity.
    """
    scores = as_scalar_array(
        scores,
        name="scores",
    ).astype(float)
    targets = as_integer_array(
        targets,
        name="targets",
    )
    thresholds = as_scalar_array(
        thresholds,
        name="thresholds",
    ).astype(float)

    validate_same_length(
        scores,
        targets,
        names=("scores", "targets"),
    )

    finite = np.isfinite(scores)
    scores = scores[finite]
    targets = targets[finite]

    positive = targets == positive_label
    positive_count = int(np.count_nonzero(positive))

    if positive_count == 0:
        raise ValueError(f"No events have positive label {positive_label}.")

    efficiency = np.full(
        len(thresholds),
        np.nan,
        dtype=float,
    )
    purity = np.full(
        len(thresholds),
        np.nan,
        dtype=float,
    )

    for index, threshold in enumerate(thresholds):
        selected = scores >= threshold

        true_positive_count = np.count_nonzero(selected & positive)
        selected_count = np.count_nonzero(selected)

        efficiency[index] = true_positive_count / positive_count

        if selected_count > 0:
            purity[index] = true_positive_count / selected_count

    return thresholds, efficiency, purity


def _as_integer_labels(
    values: ArrayLike,
    *,
    name: str,
) -> NDArray[np.int64]:
    values = as_integer_array(values, name=name)
    numeric_values = np.asarray(values)

    if not np.issubdtype(numeric_values.dtype, np.number):
        raise ValueError(f"{name} must contain integer values.")

    if not np.all(np.isfinite(numeric_values)):
        raise ValueError(f"{name} must contain finite integer values.")

    integer_values = numeric_values.astype(np.int64)

    if not np.array_equal(numeric_values, integer_values):
        raise ValueError(f"{name} must contain integer values.")

    return integer_values


def _validate_confusion_matrix(
    matrix: ArrayLike,
) -> NDArray[np.float64]:
    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square, got shape {matrix.shape}.")
    if np.any(matrix < 0.0):
        raise ValueError("matrix values must be non-negative.")

    return matrix


def _safe_diagonal_ratio(
    matrix: NDArray[np.float64],
    totals: NDArray[np.float64],
) -> NDArray[np.float64]:
    result = np.full(len(totals), np.nan, dtype=float)

    np.divide(
        np.diag(matrix),
        totals,
        out=result,
        where=totals != 0,
    )

    return result
