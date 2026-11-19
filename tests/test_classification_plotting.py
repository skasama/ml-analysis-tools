import matplotlib.pyplot as plt
import numpy as np
import pytest

from analysis_tools.plotting.classification import (
    plot_classification_efficiency,
    plot_classification_purity,
    plot_confusion_matrix,
)


def test_plot_classification_efficiency() -> None:
    figure, axis = plot_classification_efficiency(
        [50.0, 150.0],
        [0.8, 0.9],
        xlabel="True energy [MeV]",
    )

    assert axis.get_xlabel() == "True energy [MeV]"
    assert axis.get_ylabel() == "Efficiency"

    plt.close(figure)


def test_plot_classification_purity_adds_to_existing_axes() -> None:
    figure, axis = plot_classification_purity(
        [50.0, 150.0],
        [0.8, 0.9],
        label="GAT",
    )

    returned_figure, returned_axis = plot_classification_purity(
        [50.0, 150.0],
        [0.7, 0.85],
        ax=axis,
        label="fiTQun",
    )

    assert returned_figure is figure
    assert returned_axis is axis
    assert len(axis.lines) == 2

    plt.close(figure)


def test_plot_confusion_matrix() -> None:
    matrix = np.array(
        [
            [8, 2],
            [1, 9],
        ]
    )

    figure, axis = plot_confusion_matrix(
        matrix,
        class_labels=[-11, 13],
        colorbar=False,
    )

    assert axis.figure is figure
    assert axis.get_xlabel() == "Predicted class"
    assert axis.get_ylabel() == "True class"
    assert [tick.get_text() for tick in axis.get_xticklabels()] == [
        "-11",
        "13",
    ]
    assert len(axis.images) == 1
    assert len(axis.texts) == 4

    plt.close(figure)


def test_plot_confusion_matrix_normalizes_by_truth() -> None:
    figure, axis = plot_confusion_matrix(
        [
            [8, 2],
            [1, 9],
        ],
        class_labels=["electron", "muon"],
        normalize="true",
        colorbar=False,
        annotate=False,
    )

    actual = axis.images[0].get_array()

    np.testing.assert_allclose(
        actual,
        [
            [0.8, 0.2],
            [0.1, 0.9],
        ],
    )

    plt.close(figure)


def test_plot_confusion_matrix_rejects_invalid_normalization() -> None:
    with pytest.raises(ValueError, match="normalize"):
        plot_confusion_matrix(
            np.eye(2),
            class_labels=[0, 1],
            normalize="invalid",
        )


def test_plot_confusion_matrix_rejects_wrong_label_count() -> None:
    with pytest.raises(ValueError, match="Expected 2 class labels"):
        plot_confusion_matrix(
            np.eye(2),
            class_labels=[0],
        )


def test_plot_confusion_matrix_forwards_kwargs() -> None:
    figure, axis = plot_confusion_matrix(
        [
            [8, 2],
            [1, 9],
        ],
        class_labels=["Electron", "Muon"],
        colorbar=False,
        interpolation="nearest",
        aspect="equal",
        zorder=4,
    )

    image = axis.images[0]

    assert image.get_interpolation() == "nearest"
    assert image.get_zorder() == 4

    assert axis.get_aspect() == 1.0

    plt.close(figure)


def test_plot_classification_efficiency_forwards_series_kwargs() -> None:
    figure, axis = plot_classification_efficiency(
        [100.0, 200.0],
        [0.8, 0.9],
        zorder=11,
        drawstyle="steps-mid",
    )

    line = axis.lines[0]

    assert line.get_zorder() == pytest.approx(11.1)
    assert line.get_drawstyle() == "steps-mid"

    plt.close(figure)
