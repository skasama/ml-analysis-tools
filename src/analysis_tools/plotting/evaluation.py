from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from analysis_tools.analysis.evaluation import DependencyEvaluation
from analysis_tools.plotting.common import plot_dependency, plot_event_counts


def plot_evaluation(
    result: DependencyEvaluation,
    *,
    ax: Axes | None = None,
    label: str | None = None,
    color: str | None = None,
    show_counts: bool = False,
    count_ax: Axes | None = None,
    **kwargs: Any,
) -> Axes | tuple[Axes, Axes]:
    """Plot a dependency evaluation on existing or newly created axes."""
    if show_counts:
        if ax is None:
            _, axes = plt.subplots(
                2,
                1,
                sharex=True,
                figsize=(7.0, 6.0),
                gridspec_kw={
                    "height_ratios": (4.0, 1.0),
                    "hspace": 0.08,
                },
            )
            metric_ax = axes[0]
            event_ax = axes[1]
        else:
            if count_ax is None:
                raise ValueError(
                    "count_ax is required when ax is provided and show_counts=True."
                )
            metric_ax = ax
            event_ax = count_ax

        plot_dependency(
            result.centers,
            result.values,
            errors=result.errors,
            ax=metric_ax,
            label=label,
            color=color,
            **kwargs,
        )

        plot_event_counts(
            result.centers,
            result.counts,
            ax=event_ax,
            label=label,
            color=color,
        )

        event_ax.set_ylabel("Events")
        return metric_ax, event_ax

    if ax is None:
        _, ax = plt.subplots()

    plot_dependency(
        result.centers,
        result.values,
        errors=result.errors,
        ax=ax,
        label=label,
        color=color,
        **kwargs,
    )

    return ax
