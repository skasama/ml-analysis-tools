from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from analysis_tools.data.arrays import (
    as_integer_array,
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
