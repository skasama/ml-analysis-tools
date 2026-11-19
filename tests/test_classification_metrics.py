import numpy as np
import pytest

from analysis_tools.metrics.classification import (
    binned_classification_counts,
    binned_classification_metrics,
    class_metrics,
    classification_accuracy,
    classification_efficiency,
    classification_purity,
    confusion_matrix,
    predicted_labels,
    roc_auc,
    roc_curve,
    softmax,
    threshold_metrics,
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


def test_softmax() -> None:
    probabilities = softmax(
        [
            [1.0, 2.0, 3.0],
            [1000.0, 1001.0, 1002.0],
        ]
    )

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        [1.0, 1.0],
    )
    np.testing.assert_allclose(
        probabilities[0],
        probabilities[1],
    )


def test_classification_accuracy() -> None:
    actual = classification_accuracy(
        predictions=[-11, 13, 13, 11],
        targets=[-11, 11, 13, 11],
    )

    assert actual == pytest.approx(0.75)


def test_class_metrics() -> None:
    labels, efficiency, purity = class_metrics(
        predictions=[-11, 13, 13, 11],
        targets=[-11, 11, 13, 11],
        class_labels=[-11, 11, 13],
    )

    np.testing.assert_array_equal(labels, [-11, 11, 13])
    np.testing.assert_allclose(
        efficiency,
        [1.0, 0.5, 1.0],
    )
    np.testing.assert_allclose(
        purity,
        [1.0, 1.0, 0.5],
    )


def test_roc_curve() -> None:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        scores=[0.1, 0.4, 0.35, 0.8],
        targets=[0, 0, 1, 1],
        positive_label=1,
    )

    np.testing.assert_allclose(
        false_positive_rate,
        [0.0, 0.0, 0.5, 0.5, 1.0],
    )
    np.testing.assert_allclose(
        true_positive_rate,
        [0.0, 0.5, 0.5, 1.0, 1.0],
    )
    assert np.isinf(thresholds[0])


def test_roc_auc() -> None:
    actual = roc_auc(
        scores=[0.1, 0.4, 0.35, 0.8],
        targets=[0, 0, 1, 1],
        positive_label=1,
    )

    assert actual == pytest.approx(0.75)


def test_threshold_metrics() -> None:
    thresholds, efficiency, purity = threshold_metrics(
        scores=[0.1, 0.4, 0.35, 0.8],
        targets=[0, 0, 1, 1],
        thresholds=[0.0, 0.4, 0.9],
        positive_label=1,
    )

    np.testing.assert_allclose(
        thresholds,
        [0.0, 0.4, 0.9],
    )
    np.testing.assert_allclose(
        efficiency,
        [1.0, 0.5, 0.0],
    )
    np.testing.assert_allclose(
        purity[:2],
        [0.5, 0.5],
    )
    assert np.isnan(purity[2])


def test_roc_curve_rejects_missing_positive_class() -> None:
    with pytest.raises(ValueError, match="No events"):
        roc_curve(
            scores=[0.1, 0.2],
            targets=[0, 0],
            positive_label=1,
        )


def test_binned_classification_metrics() -> None:
    centers, efficiency, purity, counts = binned_classification_metrics(
        x=[0.5, 1.5, 2.5, 3.5],
        predictions=[1, 0, 1, 1],
        targets=[1, 1, 0, 1],
        edges=[0.0, 2.0, 4.0],
        class_label=1,
    )

    np.testing.assert_allclose(
        centers,
        [1.0, 3.0],
    )
    np.testing.assert_allclose(
        efficiency,
        [0.5, 1.0],
    )
    np.testing.assert_allclose(
        purity,
        [1.0, 0.5],
    )
    np.testing.assert_array_equal(
        counts,
        [2, 2],
    )


def test_binned_classification_metrics_supports_signed_labels() -> None:
    _, efficiency, purity, _ = binned_classification_metrics(
        x=[0.5, 1.5, 2.5],
        predictions=[-11, 13, -11],
        targets=[-11, -11, 13],
        edges=[0.0, 3.0],
        class_label=-11,
    )

    np.testing.assert_allclose(
        efficiency,
        [0.5],
    )
    np.testing.assert_allclose(
        purity,
        [0.5],
    )


def test_binned_classification_metrics_masks_low_statistics() -> None:
    _, efficiency, purity, counts = binned_classification_metrics(
        x=[0.5, 2.5],
        predictions=[1, 1],
        targets=[1, 0],
        edges=[0.0, 2.0, 4.0],
        class_label=1,
        minimum_events=2,
    )

    assert np.all(np.isnan(efficiency))
    assert np.all(np.isnan(purity))

    np.testing.assert_array_equal(
        counts,
        [1, 1],
    )


def test_binned_classification_metrics_returns_nan_for_empty_denominator() -> None:
    _, efficiency, purity, _ = binned_classification_metrics(
        x=[0.5, 1.5],
        predictions=[0, 0],
        targets=[0, 0],
        edges=[0.0, 2.0],
        class_label=1,
    )

    assert np.isnan(efficiency[0])
    assert np.isnan(purity[0])


def test_binned_classification_counts() -> None:
    (
        centers,
        true_positive_counts,
        true_class_counts,
        predicted_class_counts,
        event_counts,
    ) = binned_classification_counts(
        x=[0.5, 1.5, 2.5, 3.5],
        predictions=[1, 0, 1, 1],
        targets=[1, 1, 0, 1],
        edges=[0.0, 2.0, 4.0],
        class_label=1,
    )

    np.testing.assert_allclose(
        centers,
        [1.0, 3.0],
    )
    np.testing.assert_array_equal(
        true_positive_counts,
        [1, 1],
    )
    np.testing.assert_array_equal(
        true_class_counts,
        [2, 1],
    )
    np.testing.assert_array_equal(
        predicted_class_counts,
        [1, 2],
    )
    np.testing.assert_array_equal(
        event_counts,
        [2, 2],
    )
