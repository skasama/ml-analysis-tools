from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

from analysis_tools.analysis.vertex_regression import VertexRegressionResult
from analysis_tools.metrics.statistics import binned_absolute_quantile, binned_bias
from analysis_tools.plotting.common import plot_histogram, plot_series


def plot_vertex_errors(
    errors: ArrayLike,
    *,
    ax: Axes | None = None,
    bins: int | Sequence[float] = 50,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    label: str | None = None,
    color: str | None = None,
    histtype: str = "step",
    linewidth: float = 1.5,
    alpha: float | None = None,
    density: bool = False,
    figsize: tuple[float, float] = (6.4, 4.8),
) -> tuple[Figure, Axes]:
    """

    Plot a histogram of three-dimensional vertex errors.

    Parameters
    ----------
    errors : ArrayLike
        The three-dimensional vertex errors to plot.
    ax : Axes, optional
        The Matplotlib Axes to plot on. If None, a new figure and axes will be created.
    bins : int or sequence of float, optional
        The number of bins or the bin edges for the histogram. Default is 50.
    xlabel : str, optional
        The label for the x-axis. If None, a default label will be used.
    ylabel : str, optional
        The label for the y-axis. If None, a default label will be used.
    title : str, optional
        The title of the plot. If None, no title will be set.
    xlim : tuple of float, optional
        The limits for the x-axis. If None, the limits will be determined automatically.
    ylim : tuple of float, optional
        The limits for the y-axis. If None, the limits will be determined automatically.
    label : str, optional
        The label for the histogram. If None, no label will be set.
    color : str, optional
        The color of the histogram. If None, the default color will be used.
    histtype : str, optional
        The type of histogram to draw. Default is "step".
    linewidth : float, optional
        The width of the histogram lines. Default is 1.5.
    alpha : float, optional
        The transparency level of the histogram. If None,
        the default alpha will be used.
    density : bool, optional
        If True, the histogram will be normalized to form a probability density.
        Default is False.
    figsize : tuple of float, optional
        The size of the figure to create if ax is None. Default is (6.4, 4.8).

    Returns
    -------
    tuple of (Figure, Axes)
        The Matplotlib Figure and Axes objects containing the plot.
    """

    return plot_histogram(
        errors,
        ax=ax,
        bins=bins,
        default_xlabel="3D vertex error",
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        xlim=xlim,
        ylim=ylim,
        label=label,
        color=color,
        histtype=histtype,
        linewidth=linewidth,
        alpha=alpha,
        density=density,
        figsize=figsize,
    )


def plot_vertex_resolution(
    x: ArrayLike,
    resolution: ArrayLike,
    *,
    yerr: ArrayLike | None = None,
    ax: Axes | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    label: str | None = None,
    color: str | None = None,
    marker: str | None = "o",
    linestyle: str = "-",
    linewidth: float = 1.5,
    markersize: float = 4.0,
    capsize: float = 3.0,
    alpha: float | None = None,
    figsize: tuple[float, float] = (6.4, 4.8),
) -> tuple[Figure, Axes]:
    """

    Plot vertex resolution as a function of another quantity.

    Parameters
    ----------
    x : ArrayLike
        The x-axis values for the plot.
    resolution : ArrayLike
        The vertex resolution values to plot.
    yerr : ArrayLike, optional
        The error values for the resolution. If None, no error bars will be plotted.
    ax : Axes, optional
        The Matplotlib Axes to plot on. If None, a new figure and axes will be created.
    xlabel : str, optional
        The label for the x-axis. If None, a default label will be used.
    ylabel : str, optional
        The label for the y-axis. If None, a default label will be used.
    title : str, optional
        The title of the plot. If None, no title will be set.
    xlim : tuple of float, optional
        The limits for the x-axis. If None, the limits will be determined automatically.
    ylim : tuple of float, optional
        The limits for the y-axis. If None, the limits will be determined automatically.
    label : str, optional
        The label for the data series. If None, no label will be set.
    color : str, optional
        The color of the data series. If None, the default color will be used.
    marker : str, optional
        The marker style for the data points. Default is "o".
    linestyle : str, optional
        The line style for the data series. Default is "-".
    linewidth : float, optional
        The width of the lines in the plot. Default is 1.5.
    markersize : float, optional
        The size of the markers in the plot. Default is 4.0.
    capsize : float, optional
        The size of the caps on the error bars. Default is 3.0.
    alpha : float, optional
        The transparency level of the plot elements. If None,
        the default alpha will be used.
    figsize : tuple of float, optional
        The size of the figure to create if ax is None. Default is (6.4, 4.8).

    Returns
    -------
    tuple of (Figure, Axes)
        The Matplotlib Figure and Axes objects containing the plot.
    """
    return plot_series(
        x,
        resolution,
        yerr=yerr,
        ax=ax,
        xlabel=xlabel,
        ylabel=ylabel or "Vertex resolution",
        title=title,
        xlim=xlim,
        ylim=ylim,
        label=label,
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=linewidth,
        markersize=markersize,
        capsize=capsize,
        alpha=alpha,
        figsize=figsize,
    )


def plot_vertex_error_summary(
    result: VertexRegressionResult,
    *,
    bins: int | Sequence[float] = 100,
    figsize: tuple[float, float] = (15.0, 9.0),
) -> tuple[Figure, NDArray[np.object_]]:
    """Plot a summary of vertex residual and error distributions."""
    figure, axes = plt.subplots(
        2,
        3,
        figsize=figsize,
    )

    plot_data: tuple[tuple[NDArray[np.float64], str], ...] = (
        (result.x_residuals, "x residual [cm]"),
        (result.y_residuals, "y residual [cm]"),
        (result.z_residuals, "z residual [cm]"),
        (result.radius_residuals, "Radial residual [cm]"),
        (result.wall_residuals, "Wall residual [cm]"),
        (result.errors, "3D vertex error [cm]"),
    )

    for axis, (values, xlabel) in zip(
        axes.flat,
        plot_data,
        strict=True,
    ):
        plot_histogram(
            values,
            ax=axis,
            bins=bins,
            xlabel=xlabel,
            ylabel="Events",
            color="tab:blue",
            histtype="step",
            linewidth=1.5,
        )

    figure.tight_layout()

    return figure, axes


def plot_vertex_prediction(
    result: VertexRegressionResult,
    *,
    figsize: tuple[float, float] = (20.0, 4.0),
    color: str = "tab:blue",
    alpha: float = 0.3,
    marker_size: float = 5.0,
) -> tuple[Figure, NDArray[np.object_]]:
    """
    Plot predicted vertex quantities against their targets.

    Parameters
    ----------
    result : VertexRegressionResult
        The result of vertex regression analysis containing predictions and targets.
    figsize : tuple of float, optional
        The size of the figure to create. Default is (20.0, 4.0).
    color : str, optional
        The color of the scatter points. Default is "tab:blue".
    alpha : float, optional
        The transparency level of the scatter points. Default is 0.3.
    marker_size : float, optional
        The size of the scatter points. Default is 5.0.

    Returns
    -------
    tuple of (Figure, NDArray[np.object_])
        The Matplotlib Figure and an array of Axes objects containing the plots.
    """
    figure, axes = plt.subplots(
        1,
        5,
        figsize=figsize,
    )

    plot_data = (
        (
            result.targets[:, 0],
            result.predictions[:, 0],
            "True x [cm]",
            "Predicted x [cm]",
        ),
        (
            result.targets[:, 1],
            result.predictions[:, 1],
            "True y [cm]",
            "Predicted y [cm]",
        ),
        (
            result.targets[:, 2],
            result.predictions[:, 2],
            "True z [cm]",
            "Predicted z [cm]",
        ),
        (
            result.target_radius,
            result.predicted_radius,
            "True R [cm]",
            "Predicted R [cm]",
        ),
        (
            result.target_wall,
            result.predicted_wall,
            "True Wall [cm]",
            "Predicted Wall [cm]",
        ),
    )

    for axis, (targets, predictions, xlabel, ylabel) in zip(
        axes,
        plot_data,
        strict=True,
    ):
        axis.scatter(
            targets,
            predictions,
            color=color,
            alpha=alpha,
            s=marker_size,
        )

        minimum = float(min(targets.min(), predictions.min()))
        maximum = float(max(targets.max(), predictions.max()))

        axis.plot(
            [minimum, maximum],
            [minimum, maximum],
            color="black",
            linestyle="--",
            linewidth=1.0,
        )

        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_xlim(minimum, maximum)
        axis.set_ylim(minimum, maximum)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)

    figure.tight_layout()

    return figure, axes


def plot_vertex_profiles(
    result: VertexRegressionResult,
    *,
    radius_edges: ArrayLike,
    z_edges: ArrayLike,
    wall_edges: ArrayLike,
    quantile_level: float = 0.68,
    minimum_events: int = 20,
    figsize: tuple[float, float] = (15.0, 9.0),
    color: str = "tab:blue",
    marker: str = "o",
) -> tuple[Figure, NDArray[np.object_]]:
    """

    Plot binned vertex bias and resolution profiles.

    Parameters
    ----------
    result : VertexRegressionResult
        The result of the vertex regression.
    radius_edges : ArrayLike
        The bin edges for the radius.
    z_edges : ArrayLike
        The bin edges for the z-coordinate.
    wall_edges : ArrayLike
        The bin edges for the wall.
    quantile_level : float, optional
        The quantile level to use for the resolution. Default is 0.68.
    minimum_events : int, optional
        The minimum number of events required in a bin to calculate the bias
        and resolution.
        Bins with fewer events will be assigned NaN. Default is 20.
    figsize : tuple[float, float], optional
        The size of the figure. Default is (15.0, 9.0).
    color : str, optional
        The color of the scatter points and lines. Default is "tab:blue".
    marker : str, optional
        The marker to use for the scatter points. Default is "o".

    Returns
    -------
    tuple[Figure, NDArray[np.object_]]
        The Matplotlib Figure and an array of Axes objects containing the plots.
    """
    figure, axes = plt.subplots(
        2,
        3,
        figsize=figsize,
    )

    profiles = (
        (
            result.target_radius,
            result.radius_residuals,
            radius_edges,
            "True R [cm]",
        ),
        (
            result.targets[:, 2],
            result.z_residuals,
            z_edges,
            "True z [cm]",
        ),
        (
            result.target_wall,
            result.wall_residuals,
            wall_edges,
            "True Wall [cm]",
        ),
    )

    for column, (x, residuals, edges, xlabel) in enumerate(profiles):
        bias_centers, bias_values, _ = binned_bias(
            x,
            residuals,
            edges,
            minimum_events=minimum_events,
        )

        resolution_centers, resolution_values, _ = binned_absolute_quantile(
            x,
            residuals,
            edges,
            quantile_level=quantile_level,
            minimum_events=minimum_events,
        )

        plot_series(
            bias_centers,
            bias_values,
            ax=axes[0, column],
            xlabel=xlabel,
            ylabel="Bias [cm]",
            color=color,
            marker=marker,
        )
        axes[0, column].axhline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=1.0,
        )
        axes[0, column].grid(alpha=0.2)

        plot_series(
            resolution_centers,
            resolution_values,
            ax=axes[1, column],
            xlabel=xlabel,
            ylabel=f"{100.0 * quantile_level:.0f}% resolution [cm]",
            color=color,
            marker=marker,
        )
        axes[1, column].grid(alpha=0.2)

    figure.tight_layout()

    return figure, axes
