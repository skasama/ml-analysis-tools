from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from analysis_tools.plotting.common import (
    annotate_values,
    plot_dependency,
    plot_histogram,
    plot_prediction_vs_truth,
    plot_residual_distribution,
    plot_scatter,
    plot_series,
)


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


def test_plot_scatter_forwards_kwargs() -> None:
    figure, axis = plot_scatter(
        [1.0, 2.0],
        [3.0, 4.0],
        edgecolors="black",
        linewidths=0.5,
        rasterized=True,
    )

    collection = axis.collections[0]

    assert collection.get_rasterized()
    np.testing.assert_allclose(
        collection.get_linewidths(),
        [0.5],
    )

    plt.close(figure)


def test_plot_series_forwards_series_kwargs() -> None:
    figure, axis = plot_series(
        [1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0],
        zorder=8,
        dash_capstyle="round",
    )

    line = axis.lines[0]

    assert line.get_zorder() == pytest.approx(8.1)
    assert line.get_dash_capstyle() == "round"

    plt.close(figure)


def test_plot_histogram_forwards_hist_kwargs() -> None:
    figure, axis = plt.subplots()

    with patch.object(
        axis,
        "hist",
        wraps=axis.hist,
    ) as hist:
        plot_histogram(
            [1.0, 2.0, 3.0],
            ax=axis,
            bins=[0.0, 2.0, 4.0],
            hatch="//",
            linestyle=":",
        )

    hist.assert_called_once()

    kwargs = hist.call_args.kwargs

    assert kwargs["hatch"] == "//"
    assert kwargs["linestyle"] == ":"

    plt.close(figure)


def test_plot_scatter_forwards_scatter_kwargs() -> None:
    figure, axis = plot_scatter(
        [1.0, 2.0],
        [3.0, 4.0],
        rasterized=True,
        edgecolors="black",
        linewidths=0.5,
        zorder=10,
    )

    collection = axis.collections[0]

    assert collection.get_rasterized()
    assert collection.get_zorder() == 10

    np.testing.assert_allclose(
        collection.get_linewidths(),
        [0.5],
    )

    plt.close(figure)


def test_plot_prediction_vs_truth() -> None:
    predictions = np.array([1.0, 2.0, 3.0])
    targets = np.array([1.1, 1.9, 3.2])

    figure, axis = plt.subplots()

    returned_axis = plot_prediction_vs_truth(
        predictions,
        targets,
        ax=axis,
    )

    assert returned_axis is axis
    assert len(axis.collections) == 1
    assert len(axis.lines) == 1

    plt.close(figure)


def test_plot_residual_distribution() -> None:
    residuals = np.array([-1.0, 0.0, 1.0, 2.0])

    figure, axis = plt.subplots()

    returned_axis = plot_residual_distribution(
        residuals,
        ax=axis,
        bins=4,
        label="ML",
    )

    assert returned_axis is axis
    assert len(axis.patches) > 0
    assert len(axis.lines) == 1

    legend_labels = axis.get_legend_handles_labels()[1]

    assert legend_labels == ["ML, Bias: 0.5"]

    plt.close(figure)


def test_plot_residual_distribution_without_bias() -> None:
    figure, axis = plt.subplots()

    plot_residual_distribution(
        np.array([-1.0, 0.0, 1.0]),
        ax=axis,
        label="Sample",
        show_bias=False,
    )

    legend_labels = axis.get_legend_handles_labels()[1]

    assert legend_labels == ["Sample"]

    plt.close(figure)


def test_plot_residual_distribution_without_zero_line() -> None:
    figure, axis = plt.subplots()

    plot_residual_distribution(
        np.array([-1.0, 0.0, 1.0]),
        ax=axis,
        show_zero_line=False,
    )

    assert len(axis.lines) == 0

    plt.close(figure)


def test_plot_residual_distribution_rejects_multidimensional_input() -> None:
    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        plot_residual_distribution(
            np.array(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            )
        )


def test_plot_residual_distribution_rejects_nonfinite_input() -> None:
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        plot_residual_distribution(np.array([1.0, np.nan]))


def test_plot_residual_distribution_with_resolution() -> None:
    figure, axis = plt.subplots()

    plot_residual_distribution(
        np.array([-1.0, 0.0, 1.0]),
        ax=axis,
        label="ML",
        resolution=1.25,
        unit="cm",
    )

    labels = axis.get_legend_handles_labels()[1]

    assert labels == ["ML, Bias: 0 cm, Resolution: 1.25 cm"]

    plt.close(figure)


def test_plot_dependency() -> None:
    figure, axis = plt.subplots()

    returned_axis = plot_dependency(
        x=np.array([25.0, 75.0, 125.0]),
        values=np.array([10.0, 8.0, 7.0]),
        ax=axis,
        label="ML",
        color="tab:blue",
        marker="o",
    )

    assert returned_axis is axis
    assert len(axis.lines) == 1

    labels = axis.get_legend_handles_labels()[1]

    assert labels == ["ML"]

    plt.close(figure)


def test_plot_dependency_with_symmetric_errors() -> None:
    figure, axis = plt.subplots()

    returned_axis = plot_dependency(
        x=np.array([25.0, 75.0, 125.0]),
        values=np.array([10.0, 8.0, 7.0]),
        errors=np.array([1.0, 0.8, 0.7]),
        ax=axis,
        marker="o",
        capsize=2.0,
    )

    assert returned_axis is axis
    assert len(axis.containers) == 1

    plt.close(figure)


def test_plot_dependency_with_asymmetric_errors() -> None:
    figure, axis = plt.subplots()

    plot_dependency(
        x=np.array([25.0, 75.0]),
        values=np.array([10.0, 8.0]),
        errors=np.array(
            [
                [0.5, 0.4],
                [1.0, 0.8],
            ]
        ),
        ax=axis,
        marker="o",
    )

    assert len(axis.containers) == 1

    plt.close(figure)


def test_plot_dependency_ignores_nonfinite_values() -> None:
    figure, axis = plt.subplots()

    plot_dependency(
        x=np.array([25.0, 75.0, 125.0]),
        values=np.array([10.0, np.nan, 7.0]),
        ax=axis,
        marker="o",
    )

    line = axis.lines[0]

    np.testing.assert_allclose(
        line.get_xdata(),
        [25.0, 125.0],
    )

    np.testing.assert_allclose(
        line.get_ydata(),
        [10.0, 7.0],
    )

    plt.close(figure)


def test_plot_dependency_rejects_length_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="same length",
    ):
        plot_dependency(
            x=np.array([1.0, 2.0]),
            values=np.array([1.0]),
        )


def test_plot_dependency_rejects_invalid_error_shape() -> None:
    with pytest.raises(
        ValueError,
        match="shape",
    ):
        plot_dependency(
            x=np.array([1.0, 2.0]),
            values=np.array([1.0, 2.0]),
            errors=np.ones((2, 2, 2)),
        )


def test_plot_dependency_rejects_negative_errors() -> None:
    with pytest.raises(
        ValueError,
        match="negative",
    ):
        plot_dependency(
            x=np.array([1.0, 2.0]),
            values=np.array([1.0, 2.0]),
            errors=np.array([0.1, -0.2]),
        )


def test_annotate_values() -> None:
    figure, axis = plt.subplots()

    returned_axis = annotate_values(
        x=np.array([25.0, 75.0]),
        values=np.array([10.25, 8.5]),
        ax=axis,
        fmt=".2f",
    )

    assert returned_axis is axis
    assert len(axis.texts) == 2
    assert axis.texts[0].get_text() == "10.25"
    assert axis.texts[1].get_text() == "8.50"

    plt.close(figure)


def test_annotate_values_ignores_nonfinite_values() -> None:
    figure, axis = plt.subplots()

    annotate_values(
        x=np.array([25.0, 75.0, 125.0]),
        values=np.array([10.0, np.nan, 7.0]),
        ax=axis,
    )

    assert len(axis.texts) == 2

    plt.close(figure)


def test_annotate_values_rejects_length_mismatch() -> None:
    figure, axis = plt.subplots()

    with pytest.raises(
        ValueError,
        match="same length",
    ):
        annotate_values(
            x=np.array([25.0, 75.0]),
            values=np.array([10.0]),
            ax=axis,
        )

    plt.close(figure)
