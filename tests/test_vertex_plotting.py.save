import matplotlib.pyplot as plt
import numpy as np

from analysis_tools.plotting.vertex import (
    plot_vertex_errors,
    plot_vertex_resolution,
)


def test_plot_vertex_errors_creates_figure_and_axes() -> None:
    figure, axis = plot_vertex_errors(
        [10.0, 20.0, 30.0],
        bins=[0.0, 20.0, 40.0],
    )

    assert axis.figure is figure
    assert axis.get_xlabel() == "3D vertex error"
    assert axis.get_ylabel() == "Events"

    plt.close(figure)


def test_plot_vertex_errors_applies_axis_settings() -> None:
    figure, axis = plot_vertex_errors(
        [10.0, 20.0, 30.0],
        xlabel="3D vertex error [cm]",
        ylabel="Number of events",
        xlim=(0.0, 100.0),
        ylim=(0.0, 10.0),
    )

    assert axis.get_xlabel() == "3D vertex error [cm]"
    assert axis.get_ylabel() == "Number of events"
    np.testing.assert_allclose(axis.get_xlim(), [0.0, 100.0])
    np.testing.assert_allclose(axis.get_ylim(), [0.0, 10.0])

    plt.close(figure)


def test_plot_vertex_errors_adds_to_existing_axes() -> None:
    figure, axis = plt.subplots()
    axis.set_xlabel("Existing label")

    returned_figure, returned_axis = plot_vertex_errors(
        [10.0, 20.0],
        ax=axis,
        bins=[0.0, 20.0, 40.0],
        label="GAT",
    )

    plot_vertex_errors(
        [15.0, 25.0],
        ax=axis,
        bins=[0.0, 20.0, 40.0],
        label="CheRP",
    )

    assert len(axis.patches) == 2

    plt.close(figure)


def test_plot_vertex_errors_uses_density_label() -> None:
    figure, axis = plot_vertex_errors(
        [10.0, 20.0, 30.0],
        density=True,
    )

    assert axis.get_ylabel() == "Density"

    plt.close(figure)


def test_plot_vertex_resolution() -> None:
    figure, axis = plot_vertex_resolution(
        x=[50.0, 150.0, 250.0],
        resolution=[60.0, 55.0, 50.0],
        yerr=[2.0, 2.0, 3.0],
        xlabel="True energy [MeV]",
    )

    assert axis.figure is figure
    assert axis.get_xlabel() == "True energy [MeV]"
    assert axis.get_ylabel() == "Vertex resolution"

    plt.close(figure)


def test_plot_vertex_resolution_adds_to_existing_axes() -> None:
    figure, axis = plot_vertex_resolution(
        [50.0, 150.0],
        [60.0, 55.0],
        label="GAT",
    )

    returned_figure, returned_axis = plot_vertex_resolution(
        [50.0, 150.0],
        [100.0, 90.0],
        ax=axis,
        label="fiTQun",
    )

    assert returned_figure is figure
    assert returned_axis is axis
    assert len(axis.lines) == 2

    plt.close(figure)
