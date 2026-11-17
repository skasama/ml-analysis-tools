from collections.abc import Sequence

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

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
