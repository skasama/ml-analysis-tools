import matplotlib.pyplot as plt
import numpy as np

from analysis_tools.analysis.vertex_regression import (
    analyze_vertex_regression,
)
from analysis_tools.plotting.vertex import (
    plot_vertex_error_summary,
    plot_vertex_errors,
    plot_vertex_prediction,
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


def test_plot_vertex_error_summary() -> None:
    result = analyze_vertex_regression(
        predictions=[
            [3.0, 4.0, 0.0],
            [0.0, 0.0, 5.0],
        ],
        targets=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        detector_radius=10.0,
        detector_half_height=20.0,
    )

    figure, axes = plot_vertex_error_summary(
        result,
        bins=10,
    )

    assert axes.shape == (2, 3)

    expected_labels = [
        "x residual [cm]",
        "y residual [cm]",
        "z residual [cm]",
        "Radial residual [cm]",
        "Wall residual [cm]",
        "3D vertex error [cm]",
    ]

    assert [axis.get_xlabel() for axis in axes.flat] == expected_labels

    assert all(axis.get_ylabel() == "Events" for axis in axes.flat)

    assert all(len(axis.patches) == 1 for axis in axes.flat)

    plt.close(figure)


def test_plot_vertex_prediction() -> None:
    result = analyze_vertex_regression(
        predictions=[
            [3.0, 4.0, 0.0],
            [0.0, 0.0, 5.0],
        ],
        targets=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        detector_radius=10.0,
        detector_half_height=20.0,
    )

    figure, axes = plot_vertex_prediction(result)

    assert axes.shape == (5,)

    expected_labels = [
        ("True x [cm]", "Predicted x [cm]"),
        ("True y [cm]", "Predicted y [cm]"),
        ("True z [cm]", "Predicted z [cm]"),
        ("True R [cm]", "Predicted R [cm]"),
        ("True Wall [cm]", "Predicted Wall [cm]"),
    ]

    assert [(axis.get_xlabel(), axis.get_ylabel()) for axis in axes] == expected_labels

    assert all(len(axis.collections) == 1 for axis in axes)
    assert all(len(axis.lines) == 1 for axis in axes)

    plt.close(figure)
