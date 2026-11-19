from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from analysis_tools.plotting.common import plot_series


def plot_classification_efficiency(
    x: ArrayLike,
    efficiency: ArrayLike,
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

    Plot classification efficiency.

    Parameters
    ----------
    x : ArrayLike
        The x-values for the plot.
    efficiency : ArrayLike
        The classification efficiency values to plot.
    yerr : ArrayLike, optional
        The error values for the efficiency. If None, no error bars will be plotted.
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
        The color of the data series. If None, a default color will be used.
    marker : str, optional
        The marker style for the data points. Default is "o".
    linestyle : str, optional
        The line style for the data series. Default is "-".
    linewidth : float, optional
        The width of the line for the data series. Default is 1.5.
    markersize : float, optional
        The size of the markers for the data points. Default is 4.0.
    capsize : float, optional
        The size of the caps on the error bars. Default is 3.0.
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
        The Matplotlib Figure and Axes objects for the plot.
    """
    return plot_series(
        x,
        efficiency,
        yerr=yerr,
        ax=ax,
        xlabel=xlabel,
        ylabel=ylabel or "Efficiency",
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
        **series_kwargs,
    )


def plot_classification_purity(
    x: ArrayLike,
    purity: ArrayLike,
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

    Plot classification purity.

    Parameters
    ----------
    x : ArrayLike
        The x-values for the plot.
    purity : ArrayLike
        The classification purity values to plot.
    yerr : ArrayLike, optional
        The error values for the purity. If None, no error bars will be plotted.
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
        The color of the data series. If None, a default color will be used.
    marker : str, optional
        The marker style for the data points. Default is "o".
    linestyle : str, optional
        The line style for the data series. Default is "-".
    linewidth : float, optional
        The width of the line for the data series. Default is 1.5.
    markersize : float, optional
        The size of the markers for the data points. Default is 4.0.
    capsize : float, optional
        The size of the caps on the error bars. Default is 3.0.
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
        The Matplotlib Figure and Axes objects for the plot.
    """
    return plot_series(
        x,
        purity,
        yerr=yerr,
        ax=ax,
        xlabel=xlabel,
        ylabel=ylabel or "Purity",
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
        **series_kwargs,
    )


def plot_confusion_matrix(
    matrix: ArrayLike,
    class_labels: Sequence[int | str],
    *,
    ax: Axes | None = None,
    normalize: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    cmap: str = "Blues",
    colorbar: bool = True,
    annotate: bool = True,
    value_format: str | None = None,
    figsize: tuple[float, float] = (6.4, 4.8),
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """

    Plot a confusion matrix with truth on rows.

    Parameters
    ----------
    matrix : ArrayLike
        The confusion matrix to plot. Must be a square 2D array.
    class_labels : Sequence[int | str]
        The labels for the classes, in the order corresponding to the rows and columns
        of the matrix.
    ax : Axes, optional
        The Matplotlib Axes to plot on. If None, a new figure and axes will be created.
    normalize : str, optional
        If 'true', normalize by the number of true instances (rows).
        If 'predicted', normalize by the number of predicted instances (columns).
        If None, no normalization is applied.
    xlabel : str, optional
        The label for the x-axis. If None, a default label will be used.
    ylabel : str, optional
        The label for the y-axis. If None, a default label will be used.
    title : str, optional
        The title of the plot. If None, no title will be set.
    cmap : str, optional
        The colormap to use for the confusion matrix. Default is 'Blues'.
    colorbar : bool, optional
        Whether to display a colorbar. Default is True.
    annotate : bool, optional
        Whether to annotate the cells with the numeric values. Default is True.
    value_format : str, optional
        The format string for the numeric values in the cells. If None,
        a default format will be used.
    figsize : tuple of float, optional
        The size of the figure to create if ax is None. Default is (6.4, 4.8).
    **kwargs : Any
        Additional keyword arguments to pass to `imshow()` for the confusion matrix.

    Returns
    -------
    tuple[Figure, Axes]
        The Matplotlib Figure and Axes objects for the plot.
    """
    matrix = np.asarray(matrix)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square, got shape {matrix.shape}.")

    if np.any(matrix < 0):
        raise ValueError("matrix values must be non-negative.")

    if len(class_labels) != matrix.shape[0]:
        raise ValueError(
            f"Expected {matrix.shape[0]} class labels, got {len(class_labels)}."
        )

    plotted_matrix = _normalize_confusion_matrix(
        matrix,
        normalize=normalize,
    )

    created_axes = ax is None

    if ax is None:
        figure, ax = plt.subplots(figsize=figsize)
    else:
        figure = ax.figure

    image = ax.imshow(plotted_matrix, cmap=cmap, **kwargs)

    positions = np.arange(len(class_labels))
    labels = [str(label) for label in class_labels]

    ax.set_xticks(positions, labels=labels)
    ax.set_yticks(positions, labels=labels)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    elif created_axes:
        ax.set_xlabel("Predicted class")

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    elif created_axes:
        ax.set_ylabel("True class")

    if title is not None:
        ax.set_title(title)

    if annotate:
        _annotate_confusion_matrix(
            ax,
            plotted_matrix,
            image=image,
            value_format=value_format,
            normalized=normalize is not None,
        )

    if colorbar:
        figure.colorbar(image, ax=ax)

    return figure, ax


def _normalize_confusion_matrix(
    matrix: np.ndarray,
    *,
    normalize: str | None,
) -> np.ndarray:
    matrix = matrix.astype(float)

    if normalize is None:
        return matrix

    if normalize == "true":
        totals = matrix.sum(axis=1, keepdims=True)
    elif normalize == "predicted":
        totals = matrix.sum(axis=0, keepdims=True)
    else:
        raise ValueError("normalize must be None, 'true', or 'predicted'.")

    normalized = np.full(matrix.shape, np.nan, dtype=float)

    np.divide(
        matrix,
        totals,
        out=normalized,
        where=totals != 0.0,
    )

    return normalized


def _annotate_confusion_matrix(
    ax: Axes,
    matrix: np.ndarray,
    *,
    image,
    value_format: str | None,
    normalized: bool,
) -> None:
    if value_format is None:
        value_format = ".2f" if normalized else ".0f"

    finite_values = matrix[np.isfinite(matrix)]
    threshold = (
        0.5 * (finite_values.min() + finite_values.max()) if finite_values.size else 0.0
    )

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            text = "nan" if np.isnan(value) else format(value, value_format)
            color = "white" if value > threshold else "black"

            ax.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                color=color,
            )
