import numpy as np
import pytest

from analysis_tools.metrics.classification import (
    classification_efficiency,
    classification_purity,
    confusion_matrix,
    predicted_labels,
)


def test_predicted_labels() -> None:
    scores = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.1, 0.2, 0.7],
        ]
    )

    actual = predicted_labels(scores)

    np.testing.assert_array_equal(actual, [0, 1, 2])


def test_confusion_matrix() -> None:
    matrix, labels = confusion_matrix(
        predictions=[0, 1, 1, 2],
        targets=[0, 0, 1, 2],
        class_labels=[0, 1, 2],
    )

    np.testing.assert_array_equal(labels, [0, 1, 2])
    np.testing.assert_array_equal(
        matrix,
        [
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
    )


def test_confusion_matrix_accepts_signed_labels() -> None:
    matrix, labels = confusion_matrix(
        predictions=[-11, 13, 13, 11],
        targets=[-11, 11, 13, 11],
    )

    np.testing.assert_array_equal(labels, [-11, 11, 13])
    np.testing.assert_array_equal(
        matrix,
        [
            [1, 0, 0],
            [0, 1, 1],
            [0, 0, 1],
        ],
    )


def test_classification_efficiency() -> None:
    matrix = np.array(
        [
            [8, 2],
            [1, 9],
        ]
    )

    actual = classification_efficiency(matrix)

    np.testing.assert_allclose(actual, [0.8, 0.9])


def test_classification_purity() -> None:
    matrix = np.array(
        [
            [8, 2],
            [1, 9],
        ]
    )

    actual = classification_purity(matrix)

    np.testing.assert_allclose(actual, [8 / 9, 9 / 11])


def test_efficiency_returns_nan_for_empty_class() -> None:
    matrix = np.array(
        [
            [2, 0],
            [0, 0],
        ]
    )

    actual = classification_efficiency(matrix)

    assert actual[0] == 1.0
    assert np.isnan(actual[1])


def test_confusion_matrix_rejects_non_integer_labels() -> None:
    with pytest.raises(ValueError, match="integer values"):
        confusion_matrix(
            predictions=[0.0, 1.5],
            targets=[0, 1],
        )


def test_predicted_labels_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match=r"\(N, C\)"):
        predicted_labels([0.1, 0.9])
