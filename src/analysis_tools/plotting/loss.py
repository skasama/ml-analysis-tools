from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from analysis_tools.plotting.common import plot_series


def plot_loss(
    steps: ArrayLike,
    losses: ArrayLike,
    *,
    ax: Axes | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xscale: str | None = None,
    yscale: str | None = None,
    label: str | None = None,
    color: str | None = None,
    marker: str | None = None,
    linestyle: str = "-",
    linewidth: float = 1.5,
    markersize: float = 4.0,
    alpha: float | None = None,
    figsize: tuple[float, float] = (6.4, 4.8),
    **series_kwargs: Any,
) -> tuple[Figure, Axes]:
    """

    Plot loss as a function of step or epoch.

    Parameters
    ----------
    steps : ArrayLike
        The steps or epochs corresponding to the loss values.
    losses : ArrayLike
        The loss values to plot.
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
    xscale : str, optional
        The scale for the x-axis (e.g., 'linear', 'log'). If None, the default scale
        will be used.
    yscale : str, optional
        The scale for the y-axis (e.g., 'linear', 'log'). If None, the default scale
        will be used.
    label : str, optional
        The label for the data series. If None, no label will be set.
    color : str, optional
        The color of the data series. If None, the default color will be used.
    marker : str, optional
        The marker style for the data points. If None, no markers will be used.
    linestyle : str, optional
        The style of the line connecting the data points. Default is solid line ('-').
    linewidth : float, optional
        The width of the line connecting the data points. Default is 1.5.
    markersize : float, optional
        The size of the markers. Default is 4.0.
    alpha : float, optional
        The transparency level of the data series. If None, the default alpha
        will be used.
    figsize : tuple of float, optional
        The size of the figure to create if ax is None. Default is (6.4, 4.8).
    **series_kwargs : Any
        Additional keyword arguments to pass to `Axes.plot()`.

    Returns
    -------
    tuple[Figure, Axes]
        The Matplotlib Figure and Axes objects containing the plot.
    """
    created_axes = ax is None

    figure, ax = plot_series(
        steps,
        losses,
        ax=ax,
        xlabel=xlabel if xlabel is not None else ("Step" if created_axes else None),
        ylabel=ylabel if ylabel is not None else ("Loss" if created_axes else None),
        title=title,
        xlim=xlim,
        ylim=ylim,
        label=label,
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=linewidth,
        markersize=markersize,
        alpha=alpha,
        figsize=figsize,
        **series_kwargs,
    )

    if xscale is not None:
        ax.set_xscale(xscale)

    if yscale is not None:
        ax.set_yscale(yscale)

    return figure, ax
