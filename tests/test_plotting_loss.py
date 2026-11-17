import matplotlib.pyplot as plt
import numpy as np

from analysis_tools.plotting.loss import plot_loss


def test_plot_loss_creates_figure_and_axes() -> None:
    figure, axis = plot_loss(
        steps=[0, 100, 200],
        losses=[1.0, 0.5, 0.25],
    )

    assert axis.figure is figure
    assert axis.get_xlabel() == "Step"
    assert axis.get_ylabel() == "Loss"
    assert len(axis.lines) == 1

    plt.close(figure)


def test_plot_loss_adds_to_existing_axes() -> None:
    figure, axis = plot_loss(
        [0, 100, 200],
        [1.0, 0.5, 0.25],
        label="Training",
    )

    returned_figure, returned_axis = plot_loss(
        [0, 100, 200],
        [1.2, 0.7, 0.4],
        ax=axis,
        label="Validation",
    )

    assert returned_figure is figure
    assert returned_axis is axis
    assert len(axis.lines) == 2

    plt.close(figure)


def test_plot_loss_applies_log_scale() -> None:
    figure, axis = plot_loss(
        [1, 10, 100],
        [1.0, 0.1, 0.01],
        xscale="log",
        yscale="log",
    )

    assert axis.get_xscale() == "log"
    assert axis.get_yscale() == "log"

    plt.close(figure)


def test_plot_loss_preserves_existing_labels() -> None:
    figure, axis = plt.subplots()
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation loss")

    plot_loss(
        [0, 1, 2],
        [1.0, 0.8, 0.6],
        ax=axis,
    )

    assert axis.get_xlabel() == "Epoch"
    assert axis.get_ylabel() == "Validation loss"

    plt.close(figure)


def test_plot_loss_applies_axis_settings() -> None:
    figure, axis = plot_loss(
        [0, 100, 200],
        [1.0, 0.5, 0.25],
        xlabel="Training step",
        ylabel="Cross-entropy loss",
        xlim=(0.0, 250.0),
        ylim=(0.0, 1.5),
    )

    assert axis.get_xlabel() == "Training step"
    assert axis.get_ylabel() == "Cross-entropy loss"
    np.testing.assert_allclose(axis.get_xlim(), [0.0, 250.0])
    np.testing.assert_allclose(axis.get_ylim(), [0.0, 1.5])

    plt.close(figure)
