from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from analysis_tools.data.arrays import as_integer_array, validate_same_length
from analysis_tools.metrics.classification import predicted_labels, softmax


@dataclass(frozen=True)
class ClassificationQuantities:
    """Event-level quantities for classification."""

    scores: NDArray[np.float64]
    probabilities: NDArray[np.float64]
    predicted_labels: NDArray[np.int64]
    target_labels: NDArray[np.int64]
    correct_mask: NDArray[np.bool_]

    @property
    def event_count(self) -> int:
        return len(self.target_labels)

    def select(
        self,
        selection: ArrayLike,
    ) -> "ClassificationQuantities":
        return ClassificationQuantities(
            scores=self.scores[selection],
            probabilities=self.probabilities[selection],
            predicted_labels=self.predicted_labels[selection],
            target_labels=self.target_labels[selection],
            correct_mask=self.correct_mask[selection],
        )


def calculate_classification_quantities(
    scores: ArrayLike,
    targets: ArrayLike,
    *,
    probabilities: ArrayLike | None = None,
    class_labels: Sequence[int] | None = None,
) -> ClassificationQuantities:
    """Calculate event-level classification quantities.

    Probabilities default to a softmax of the scores. Supply probabilities
    explicitly when the model uses another probability transformation.
    """
    score_array = np.asarray(scores, dtype=float)
    target_array = as_integer_array(targets, name="targets")

    if score_array.ndim != 2:
        raise ValueError(f"scores must have shape (N, C), got {score_array.shape}.")

    validate_same_length(
        score_array,
        target_array,
        names=("scores", "targets"),
    )

    if probabilities is None:
        probability_array = softmax(score_array)
    else:
        probability_array = np.asarray(
            probabilities,
            dtype=float,
        )

        if probability_array.shape != score_array.shape:
            raise ValueError("probabilities must have the same shape as scores.")

        if not np.all(np.isfinite(probability_array)):
            raise ValueError("probabilities must contain only finite values.")

        if np.any((probability_array < 0.0) | (probability_array > 1.0)):
            raise ValueError("probabilities must be in the range [0, 1].")

        if not np.allclose(
            probability_array.sum(axis=1),
            1.0,
        ):
            raise ValueError("probability rows must sum to 1.")

    predicted_indices = predicted_labels(score_array)

    if class_labels is None:
        predicted_array = predicted_indices
    else:
        labels = as_integer_array(
            class_labels,
            name="class_labels",
        )

        if len(labels) != score_array.shape[1]:
            raise ValueError("class_labels must contain one label per score column.")

        if len(np.unique(labels)) != len(labels):
            raise ValueError("class_labels must be unique.")

        unknown_targets = np.setdiff1d(target_array, labels)

        if len(unknown_targets):
            raise ValueError(
                "targets contain labels missing from class_labels: "
                f"{unknown_targets.tolist()}."
            )

        predicted_array = labels[predicted_indices]

    return ClassificationQuantities(
        scores=score_array,
        probabilities=probability_array,
        predicted_labels=predicted_array,
        target_labels=target_array,
        correct_mask=predicted_array == target_array,
    )
