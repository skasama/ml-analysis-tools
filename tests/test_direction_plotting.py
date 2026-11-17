import matplotlib.pyplot as plt
import numpy as np

from analysis_tools.plotting.direction import (
    plot_direction_errors,
    plot_direction_resolution,
)


def test_plot_direction_errors_creates_figure_and_axes() -> None:
    figure, axis = plot_direction_errors(
        [1.0, 2.0, 3.0],
        bins=[0.0, 2.0, 4.0],
    )

    assert axis.figure is figure
    assert axis.get_xlabel() == "Direction error [degree]"
    assert axis.get_ylabel() == "Events"
    assert len(axis.patches) == 1

    plt.close(figure)


def test_plot_direction_errors_applies_axis_settings() -> None:
    figure, axis = plot_direction_errors(
        [1.0, 2.0, 3.0],
        xlabel="Opening-angle error [degree]",
        ylabel="Number of events",
        xlim=(0.0, 20.0),
        ylim=(0.0, 10.0),
    )

    assert axis.get_xlabel() == "Opening-angle error [degree]"
    assert axis.get_ylabel() == "Number of events"
    np.testing.assert_allclose(axis.get_xlim(), [0.0, 20.0])
    np.testing.assert_allclose(axis.get_ylim(), [0.0, 10.0])

    plt.close(figure)


def test_plot_direction_errors_adds_to_existing_axes() -> None:
    figure, axis = plot_direction_errors(
        [1.0, 2.0],
        bins=[0.0, 2.0, 4.0],
        label="GAT",
    )

    returned_figure, returned_axis = plot_direction_errors(
        [1.5, 2.5],
        ax=axis,
        bins=[0.0, 2.0, 4.0],
        label="CheRP",
    )

    assert returned_figure is figure
    assert returned_axis is axis
    assert len(axis.patches) == 2

    plt.close(figure)


def test_plot_direction_errors_uses_density_label() -> None:
    figure, axis = plot_direction_errors(
        [1.0, 2.0, 3.0],
        density=True,
    )

    assert axis.get_ylabel() == "Density"

    plt.close(figure)


def test_plot_direction_resolution_adds_to_existing_axes() -> None:
    figure, axis = plot_direction_resolution(
        [50.0, 150.0],
        [5.0, 4.0],
        label="GAT",
    )

    returned_figure, returned_axis = plot_direction_resolution(
        [50.0, 150.0],
        [8.0, 7.0],
        ax=axis,
        label="fiTQun",
    )

    assert returned_figure is figure
    assert returned_axis is axis
    assert axis.get_ylabel() == "Direction resolution [degree]"
    assert len(axis.lines) == 2

    plt.close(figure)
