from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike


def plot_histogram(
    values: ArrayLike,
    *,
    ax: Axes | None = None,
    bins: int | Sequence[float] = 50,
    default_xlabel: str | None = None,
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
    **hist_kwargs: Any,
) -> tuple[Figure, Axes]:
    """

    Plot a histogram on a new or existing Axes.

    Parameters
    ----------
    values : ArrayLike
        The values to plot in the histogram.
    ax : Axes, optional
        The Matplotlib Axes to plot on. If None, a new figure and axes
        will be created.
    bins : int or sequence of float, optional
        The number of bins or the bin edges for the histogram. Default is 50.
    default_xlabel : str, optional
        The default label for the x-axis if xlabel is not provided. If None,
        no default label will be set.
    xlabel : str, optional
        The label for the x-axis. If None, the default label will be used.
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
    **hist_kwargs : Any
        Additional keyword arguments to pass to `Axes.hist()`.

    Returns
    -------
    tuple[Figure, Axes]
        The Matplotlib Figure and Axes objects containing the histogram.
    """
    created_axes = ax is None

    if ax is None:
        figure, ax = plt.subplots(figsize=figsize)
    else:
        figure = ax.figure

    ax.hist(
        values,
        bins=bins,
        label=label,
        color=color,
        histtype=histtype,
        linewidth=linewidth,
        alpha=alpha,
        density=density,
        **hist_kwargs,
    )

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    elif created_axes and default_xlabel is not None:
        ax.set_xlabel(default_xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    elif created_axes:
        ax.set_ylabel("Density" if density else "Events")

    if title is not None:
        ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    return figure, ax


def plot_series(
    x: ArrayLike,
    y: ArrayLike,
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
    **series_kwargs: Any,
) -> tuple[Figure, Axes]:
    """

    Plot a data series with optional vertical uncertainties.


    Parameters
    ----------
    x : ArrayLike
        The x-coordinates of the data points.
    y : ArrayLike
        The y-coordinates of the data points.
    yerr : ArrayLike, optional
        The vertical uncertainties of the data points.
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
        The width of the line for the data series. Default is 1.5.
    markersize : float, optional
        The size of the markers for the data points. Default is 4.0.
    capsize : float, optional
        The size of the error bar caps. Default is 3.0.
    alpha : float, optional
        The transparency level of the data series. If None,
        the default alpha will be used.
    figsize : tuple of float, optional
        The size of the figure to create if ax is None. Default is (6.4, 4.8).
    **series_kwargs : Any
        Additional keyword arguments to pass to `Axes.errorbar()`.

    Returns
    -------
    tuple[Figure, Axes]
        The Matplotlib Figure and Axes objects containing the plot.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional.")

    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    if yerr is not None:
        yerr = np.asarray(yerr)

        if yerr.ndim != 1 or len(yerr) != len(y):
            raise ValueError("yerr must be one-dimensional and match y.")

    if ax is None:
        figure, ax = plt.subplots(figsize=figsize)
    else:
        figure = ax.figure

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        label=label,
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=linewidth,
        markersize=markersize,
        capsize=capsize,
        alpha=alpha,
        **series_kwargs,
    )

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    return figure, ax


def plot_scatter(
    x: ArrayLike,
    y: ArrayLike,
    *,
    ax: Axes | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    label: str | None = None,
    color: str | None = None,
    marker: str = "o",
    marker_size: float = 5.0,
    alpha: float | None = None,
    figsize: tuple[float, float] = (6.4, 4.8),
    **scatter_kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot a scatter series on a new or existing Axes."""
    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional.")

    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    if ax is None:
        figure, ax = plt.subplots(figsize=figsize)
    else:
        figure = ax.figure

    ax.scatter(
        x,
        y,
        label=label,
        color=color,
        marker=marker,
        s=marker_size,
        alpha=alpha,
        **scatter_kwargs,
    )

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    return figure, ax
