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


def plot_prediction_vs_truth(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    add_diagonal: bool = True,
    **kwargs,
) -> plt.Axes:
    """

    Plot predictions against targets as a scatter plot.

    Parameters
    ----------
    predictions : np.ndarray
        The predicted values, shape (N,).
    targets : np.ndarray
        The true values, shape (N,).
    ax : plt.Axes, optional
        The Matplotlib Axes to plot on. If None, a new figure and axes will be created.
    add_diagonal : bool, optional
        Whether to add a diagonal line representing perfect predictions.
        Default is True.
    **kwargs : Any
        Additional keyword arguments to pass to `Axes.scatter()`.

    Returns
    -------
    plt.Axes
        The Matplotlib Axes object containing the scatter plot.
    """
    predictions = np.asarray(
        predictions,
        dtype=float,
    )
    targets = np.asarray(
        targets,
        dtype=float,
    )

    if predictions.ndim != 1:
        raise ValueError("predictions must be one-dimensional.")

    if targets.ndim != 1:
        raise ValueError("targets must be one-dimensional.")

    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same number of events.")

    if ax is None:
        _, ax = plt.subplots()

    scatter_options = {
        "s": 4.0,
        "alpha": 0.3,
        "edgecolors": "none",
        "rasterized": True,
    }
    scatter_options.update(kwargs)

    ax.scatter(
        targets,
        predictions,
        **scatter_options,
    )

    if add_diagonal:
        minimum = min(
            np.min(targets),
            np.min(predictions),
        )
        maximum = max(
            np.max(targets),
            np.max(predictions),
        )

        ax.plot(
            [minimum, maximum],
            [minimum, maximum],
            color="black",
            linestyle="--",
            linewidth=1.2,
        )

    return ax


def plot_residual_distribution(
    residuals: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    bins: int | np.ndarray = 100,
    label: str | None = None,
    show_zero_line: bool = True,
    show_bias: bool = True,
    bias: float | None = None,
    resolution: float | None = None,
    unit: str | None = None,
    **kwargs,
) -> plt.Axes:
    """

    Plot a one-dimensional residual distribution.

    Parameters
    ----------
    residuals : np.ndarray
        The residuals to plot, shape (N,).
    ax : plt.Axes, optional
        The Matplotlib Axes to plot on. If None, a new figure and axes will be created.
    bins : int or np.ndarray, optional
        The number of bins or the bin edges for the histogram. Default is 100.
    label : str, optional
        The label for the histogram. If None, no label will be set.
    show_zero_line : bool, optional
        Whether to show a vertical line at zero. Default is True.
    show_bias : bool, optional
        Whether to show the bias in the legend. Default is True.
    bias : float, optional
        The bias value to display in the legend. If None, the bias will be calculated
          from the residuals.
    resolution : float, optional
        The resolution value to display in the legend. If None, no resolution
        will be displayed.
    unit : str, optional
        The unit of the residuals to display in the legend. If None, no unit
        will be displayed.
    **kwargs : Any
        Additional keyword arguments to pass to `Axes.hist()`.

    Returns
    -------
    plt.Axes
        The Matplotlib Axes object containing the histogram.
    """
    residuals = np.asarray(
        residuals,
        dtype=float,
    )

    if residuals.ndim != 1:
        raise ValueError("residuals must be one-dimensional.")

    if not np.all(np.isfinite(residuals)):
        raise ValueError("residuals must contain only finite values.")

    if ax is None:
        _, ax = plt.subplots()

    value_unit = f" {unit}" if unit is not None else ""

    label_parts = []

    if label is not None:
        label_parts.append(label)

    if show_bias:
        if bias is None:
            bias = float(np.mean(residuals))

        label_parts.append(f"Bias: {bias:.3g}{value_unit}")

    if resolution is not None:
        label_parts.append(f"Resolution: {resolution:.3g}{value_unit}")

    plot_label = ", ".join(label_parts) if label_parts else None

    ax.hist(
        residuals,
        bins=bins,
        histtype="step",
        density=True,
        label=plot_label,
        **kwargs,
    )

    if show_zero_line:
        ax.axvline(
            0.0,
            color="black",
            linestyle=":",
            linewidth=1.0,
        )

    return ax


def plot_dependency(
    x: np.ndarray,
    values: np.ndarray,
    *,
    errors: np.ndarray | None = None,
    ax: plt.Axes | None = None,
    label: str | None = None,
    **kwargs,
) -> plt.Axes:
    """

    Plot values as a function of another quantity.

    Parameters
    ----------
    x : np.ndarray
        The x-coordinates of the data points, shape (N,).
    values : np.ndarray
        The y-coordinates of the data points, shape (N,).
    errors : np.ndarray, optional
        The vertical uncertainties of the data points. Can be either shape (N,)
        for symmetric errors
        or shape (2, N) for asymmetric errors. If None, no error bars will be plotted.
        Default is None.
    ax : plt.Axes, optional
        The Matplotlib Axes to plot on. If None, a new figure and axes will be created.
    label : str, optional
        The label for the data series. If None, no label will be set.
    **kwargs : Any
        Additional keyword arguments to pass to `Axes.errorbar()` or `Axes.plot()`.

    Returns
    -------
    plt.Axes
        The Matplotlib Axes object containing the plot.
    """
    x = np.asarray(
        x,
        dtype=float,
    )
    values = np.asarray(
        values,
        dtype=float,
    )

    if x.ndim != 1:
        raise ValueError("x must be one-dimensional.")

    if values.ndim != 1:
        raise ValueError("values must be one-dimensional.")

    if len(x) != len(values):
        raise ValueError("x and values must have the same length.")

    if errors is not None:
        errors = np.asarray(
            errors,
            dtype=float,
        )

        valid_error_shape = errors.shape == values.shape or errors.shape == (
            2,
            len(values),
        )

        if not valid_error_shape:
            raise ValueError("errors must have shape (N,) or (2, N).")

        if np.any(errors < 0.0):
            raise ValueError("errors must not contain negative values.")

    if ax is None:
        _, ax = plt.subplots()

    valid = np.isfinite(x) & np.isfinite(values)

    if errors is not None:
        if errors.ndim == 1:
            valid &= np.isfinite(errors)
            selected_errors = errors[valid]
        else:
            valid &= np.all(
                np.isfinite(errors),
                axis=0,
            )
            selected_errors = errors[:, valid]

        ax.errorbar(
            x[valid],
            values[valid],
            yerr=selected_errors,
            label=label,
            **kwargs,
        )

    else:
        ax.plot(
            x[valid],
            values[valid],
            label=label,
            **kwargs,
        )

    return ax


def plot_event_counts(
    x: np.ndarray,
    counts: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    label: str | None = None,
    width: float | None = None,
    **kwargs,
) -> plt.Axes:
    """

    Plot event counts as bars.

    Parameters
    ----------
    x : np.ndarray
        The x-coordinates of the bars, shape (N,).
    counts : np.ndarray
        The event counts for each bar, shape (N,).
    ax : plt.Axes, optional
        The Matplotlib Axes to plot on. If None, a new figure and axes will be created.
    label : str, optional
        The label for the data series. If None, no label will be set.
    width : float, optional
        The width of the bars. If None, the width will be automatically determined.
    **kwargs : Any
        Additional keyword arguments to pass to `Axes.bar()`.

    Returns
    -------
    plt.Axes
        The Matplotlib Axes object containing the plot.
    """
    x = np.asarray(
        x,
        dtype=float,
    )
    counts = np.asarray(
        counts,
    )

    if x.ndim != 1:
        raise ValueError("x must be one-dimensional.")

    if counts.ndim != 1:
        raise ValueError("counts must be one-dimensional.")

    if len(x) != len(counts):
        raise ValueError("x and counts must have the same length.")

    if np.any(counts < 0):
        raise ValueError("counts must not contain negative values.")

    if ax is None:
        _, ax = plt.subplots()

    if width is None:
        if len(x) > 1:
            width = 0.8 * float(np.min(np.diff(x)))
        else:
            width = 0.8

    valid = np.isfinite(x) & np.isfinite(counts)

    ax.bar(
        x[valid],
        counts[valid],
        width=width,
        label=label,
        **kwargs,
    )

    return ax


def annotate_values(
    x: np.ndarray,
    values: np.ndarray,
    *,
    ax: plt.Axes,
    fmt: str = ".3g",
    offset_points: float = 4.0,
    **kwargs,
) -> plt.Axes:
    """

    Annotate finite values at their corresponding coordinates.

    Parameters
    ----------
    x : np.ndarray
        The x-coordinates of the values to annotate, shape (N,).
    values : np.ndarray
        The values to annotate, shape (N,).
    ax : plt.Axes
        The Matplotlib Axes to annotate.
    fmt : str, optional
        The format string for the annotated values. Default is ".3g".
    offset_points : float, optional
        The offset in points for the annotation text. Default is 4.0.
    **kwargs : Any
        Additional keyword arguments to pass to `Axes.annotate()`.

    Returns
    -------
    plt.Axes
        The Matplotlib Axes object containing the annotations.
    """
    x = np.asarray(
        x,
        dtype=float,
    )
    values = np.asarray(
        values,
        dtype=float,
    )

    if x.ndim != 1:
        raise ValueError("x must be one-dimensional.")

    if values.ndim != 1:
        raise ValueError("values must be one-dimensional.")

    if len(x) != len(values):
        raise ValueError("x and values must have the same length.")

    text_options = {
        "ha": "center",
        "va": "bottom",
        "fontsize": 8,
    }
    text_options.update(kwargs)

    valid = np.isfinite(x) & np.isfinite(values)

    for index in np.flatnonzero(valid):
        ax.annotate(
            format(
                values[index],
                fmt,
            ),
            xy=(
                x[index],
                values[index],
            ),
            xytext=(
                0.0,
                offset_points,
            ),
            textcoords="offset points",
            **text_options,
        )

    return ax
