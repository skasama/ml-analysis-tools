import matplotlib.pyplot as plt
import numpy as np
import pytest

from analysis_tools.plotting.energy import (
    plot_energy_bias,
    plot_energy_residuals,
    plot_energy_residuals_ax,
    plot_energy_resolution,
)


def test_plot_energy_residuals_ax_returns_existing_axes() -> None:
    figure, axis = plt.subplots()

    returned = plot_energy_residuals_ax(
        [-10.0, 0.0, 10.0],
        ax=axis,
    )

    assert returned is axis
    assert axis.figure is figure

    plt.close(figure)


def test_plot_energy_residuals_creates_figure_and_axes() -> None:
    figure, axis = plot_energy_residuals(
        [-10.0, 0.0, 10.0],
        bins=[-20.0, 0.0, 20.0],
    )

    assert axis.figure is figure
    assert axis.get_xlabel() == "Energy residual"
    assert axis.get_ylabel() == "Events"
    assert len(axis.patches) == 1

    plt.close(figure)


def test_plot_energy_residuals_applies_axis_settings() -> None:
    figure, axis = plot_energy_residuals(
        [-10.0, 0.0, 10.0],
        xlabel="Energy residual [MeV]",
        ylabel="Number of events",
        xlim=(-100.0, 100.0),
        ylim=(0.0, 20.0),
    )

    assert axis.get_xlabel() == "Energy residual [MeV]"
    assert axis.get_ylabel() == "Number of events"
    np.testing.assert_allclose(axis.get_xlim(), [-100.0, 100.0])
    np.testing.assert_allclose(axis.get_ylim(), [0.0, 20.0])

    plt.close(figure)


def test_plot_energy_residuals_adds_to_existing_axes() -> None:
    figure, axis = plot_energy_residuals(
        [-10.0, 0.0, 10.0],
        bins=[-20.0, 0.0, 20.0],
        label="GAT",
    )

    returned_figure, returned_axis = plot_energy_residuals(
        [-5.0, 5.0, 15.0],
        ax=axis,
        bins=[-20.0, 0.0, 20.0],
        label="CheRP",
    )

    assert returned_figure is figure
    assert returned_axis is axis
    assert len(axis.patches) == 2

    plt.close(figure)


def test_plot_energy_residuals_uses_density_label() -> None:
    figure, axis = plot_energy_residuals(
        [-10.0, 0.0, 10.0],
        density=True,
    )

    assert axis.get_ylabel() == "Density"

    plt.close(figure)


def test_plot_energy_bias_adds_to_existing_axes() -> None:
    figure, axis = plot_energy_bias(
        [50.0, 150.0],
        [0.05, -0.02],
        label="GAT",
    )

    returned_figure, returned_axis = plot_energy_bias(
        [50.0, 150.0],
        [0.10, 0.04],
        ax=axis,
        label="fiTQun",
    )

    assert returned_figure is figure
    assert returned_axis is axis
    assert axis.get_ylabel() == "Energy bias"
    assert len(axis.lines) == 2

    plt.close(figure)


def test_plot_energy_resolution() -> None:
    figure, axis = plot_energy_resolution(
        [50.0, 150.0],
        [0.20, 0.15],
        yerr=[0.01, 0.02],
        xlabel="True energy [MeV]",
    )

    assert axis.get_xlabel() == "True energy [MeV]"
    assert axis.get_ylabel() == "Energy resolution"

    plt.close(figure)


def test_plot_energy_resolution_forwards_series_kwargs() -> None:
    figure, axis = plot_energy_resolution(
        [100.0, 200.0, 300.0],
        [0.20, 0.15, 0.10],
        zorder=9,
        drawstyle="steps-mid",
    )

    line = axis.lines[0]

    assert line.get_zorder() == pytest.approx(9.1)
    assert line.get_drawstyle() == "steps-mid"

    plt.close(figure)
