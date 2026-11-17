import matplotlib.pyplot as plt
import pytest

from analysis_tools.plotting.common import plot_histogram, plot_series


def test_plot_histogram_creates_figure_and_axes() -> None:
    figure, axis = plot_histogram(
        [1.0, 2.0, 3.0],
        bins=[0.0, 2.0, 4.0],
        default_xlabel="Value",
    )

    assert axis.figure is figure
    assert axis.get_xlabel() == "Value"
    assert axis.get_ylabel() == "Events"
    assert len(axis.patches) == 1

    plt.close(figure)


def test_plot_histogram_preserves_existing_axis_settings() -> None:
    figure, axis = plt.subplots()
    axis.set_xlabel("Existing label")
    axis.set_xlim(0.0, 10.0)

    returned_figure, returned_axis = plot_histogram(
        [1.0, 2.0, 3.0],
        ax=axis,
        default_xlabel="Default label",
    )

    assert returned_figure is figure
    assert returned_axis is axis
    assert axis.get_xlabel() == "Existing label"
    assert axis.get_xlim() == (0.0, 10.0)

    plt.close(figure)


def test_plot_histogram_allows_explicit_axis_override() -> None:
    figure, axis = plt.subplots()
    axis.set_xlabel("Existing label")

    plot_histogram(
        [1.0, 2.0, 3.0],
        ax=axis,
        xlabel="New label",
        xlim=(0.0, 5.0),
    )

    assert axis.get_xlabel() == "New label"
    assert axis.get_xlim() == (0.0, 5.0)

    plt.close(figure)


def test_plot_series_creates_figure_and_axes() -> None:
    figure, axis = plot_series(
        [50.0, 150.0],
        [20.0, 30.0],
        xlabel="True energy [MeV]",
        ylabel="Resolution [%]",
    )

    assert axis.figure is figure
    assert axis.get_xlabel() == "True energy [MeV]"
    assert axis.get_ylabel() == "Resolution [%]"

    plt.close(figure)


def test_plot_series_adds_to_existing_axes() -> None:
    figure, axis = plot_series(
        [1.0, 2.0],
        [3.0, 4.0],
        label="GAT",
    )

    returned_figure, returned_axis = plot_series(
        [1.0, 2.0],
        [2.0, 3.0],
        ax=axis,
        label="CheRP",
    )

    assert returned_figure is figure
    assert returned_axis is axis
    assert len(axis.lines) == 2

    plt.close(figure)


def test_plot_series_accepts_uncertainties() -> None:
    figure, axis = plot_series(
        [1.0, 2.0],
        [3.0, 4.0],
        yerr=[0.1, 0.2],
    )

    assert len(axis.containers) == 1

    plt.close(figure)


def test_plot_series_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        plot_series(
            [1.0, 2.0],
            [3.0],
        )
