from contextlib import nullcontext
from typing import Any

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray


def plot_event_2d(
    values: ArrayLike,
    hit_coordinates: ArrayLike,
    *,
    all_pmt_coordinates: ArrayLike | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    color_label: str = "Value",
    cmap: str = "plasma",
    norm: colors.Normalize | None = None,
    show_zero: bool = False,
    background_color: str = "0.85",
    background_size: float = 8.0,
    marker_size: float = 12.0,
    alpha: float = 1.0,
    colorbar: bool = True,
    equal_aspect: bool = True,
    show_axes: bool = False,
    style: str | None = None,
    figsize: tuple[float, float] = (10.0, 8.0),
) -> tuple[Figure, Axes]:
    """Plot a two-dimensional event display.

    Parameters
    ----------
    values
        Values associated with hit PMTs, such as charge or time.
    hit_coordinates
        Two-dimensional coordinates with shape (N, 2).
    all_pmt_coordinates
        Optional coordinates of all PMTs, shown as a background.
    ax
        Existing Matplotlib Axes. A new Figure is created when omitted.
    show_zero
        If False, hits with zero values are not displayed.
    """
    values_array = _as_values(
        values,
        name="values",
    )
    hit_coordinates_array = _as_coordinates(
        hit_coordinates,
        dimensions=2,
        name="hit_coordinates",
    )

    _validate_same_length(
        values_array,
        hit_coordinates_array,
        first_name="values",
        second_name="hit_coordinates",
    )

    background_coordinates = None

    if all_pmt_coordinates is not None:
        background_coordinates = _as_coordinates(
            all_pmt_coordinates,
            dimensions=2,
            name="all_pmt_coordinates",
        )

    selected = np.isfinite(values_array)
    selected &= np.all(
        np.isfinite(hit_coordinates_array),
        axis=1,
    )

    if not show_zero:
        selected &= values_array != 0.0

    plot_values = values_array[selected]
    plot_coordinates = hit_coordinates_array[selected]

    if norm is None:
        norm = _default_norm(plot_values)

    style_context = plt.style.context(style) if style is not None else nullcontext()

    with style_context:
        if ax is None:
            figure, ax = plt.subplots(
                figsize=figsize,
            )
        else:
            figure = ax.figure

        if background_coordinates is not None:
            finite_background = np.all(
                np.isfinite(background_coordinates),
                axis=1,
            )
            background_coordinates = background_coordinates[finite_background]

            ax.scatter(
                background_coordinates[:, 0],
                background_coordinates[:, 1],
                color=background_color,
                s=background_size,
                alpha=0.45,
                linewidths=0.0,
                label="All PMTs",
            )

        hit_collection = None

        if len(plot_values) > 0:
            hit_collection = ax.scatter(
                plot_coordinates[:, 0],
                plot_coordinates[:, 1],
                c=plot_values,
                s=marker_size,
                cmap=cmap,
                norm=norm,
                alpha=alpha,
                linewidths=0.0,
            )

        if colorbar and hit_collection is not None:
            figure.colorbar(
                hit_collection,
                ax=ax,
                pad=0.02,
                label=color_label,
            )

        if title is not None:
            ax.set_title(title)

        if equal_aspect:
            ax.set_aspect(
                "equal",
                adjustable="box",
            )

        if show_axes:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(alpha=0.2)
        else:
            ax.set_axis_off()

    return figure, ax


def plot_event_3d(
    values: ArrayLike,
    hit_coordinates: ArrayLike,
    *,
    all_pmt_coordinates: ArrayLike | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    color_label: str = "Value",
    cmap: str = "plasma",
    norm: colors.Normalize | None = None,
    show_zero: bool = False,
    background_color: str = "0.55",
    background_size: float = 2.0,
    background_alpha: float = 0.035,
    marker_size: float = 22.0,
    alpha: float = 0.95,
    colorbar: bool = True,
    view_elevation: float = 25.0,
    view_azimuth: float = -60.0,
    vertical_axis: str = "z",
    show_axes: bool = True,
    show_grid: bool = True,
    show_background_panes: bool = True,
    equal_aspect: bool = True,
    axis_labels: tuple[str, str, str] = ("x", "y", "z"),
    pane_color: tuple[float, float, float, float] = (
        0.95,
        0.95,
        0.95,
        0.35,
    ),
    style: str | None = None,
    figsize: tuple[float, float] = (10.0, 8.0),
) -> tuple[Figure, Axes]:
    """Plot a three-dimensional PMT event display.

    The default display includes x, y, and z axes, grid lines, and
    lightly colored background panes to make the detector orientation
    clear.
    """
    values_array = _as_values(
        values,
        name="values",
    )
    hit_coordinates_array = _as_coordinates(
        hit_coordinates,
        dimensions=3,
        name="hit_coordinates",
    )

    _validate_same_length(
        values_array,
        hit_coordinates_array,
        first_name="values",
        second_name="hit_coordinates",
    )

    background_coordinates = None

    if all_pmt_coordinates is not None:
        background_coordinates = _as_coordinates(
            all_pmt_coordinates,
            dimensions=3,
            name="all_pmt_coordinates",
        )

    selected = np.isfinite(values_array)
    selected &= np.all(
        np.isfinite(hit_coordinates_array),
        axis=1,
    )

    if not show_zero:
        selected &= values_array != 0.0

    plot_values = values_array[selected]
    plot_coordinates = hit_coordinates_array[selected]

    if norm is None:
        norm = _default_norm(plot_values)

    style_context = plt.style.context(style) if style is not None else nullcontext()

    with style_context:
        if ax is None:
            figure = plt.figure(
                figsize=figsize,
            )
            ax = figure.add_subplot(
                111,
                projection="3d",
            )
        else:
            figure = ax.figure

            if (
                getattr(
                    ax,
                    "name",
                    None,
                )
                != "3d"
            ):
                raise TypeError("ax must be a three-dimensional Matplotlib Axes.")

        ax.view_init(
            elev=view_elevation,
            azim=view_azimuth,
            vertical_axis=vertical_axis,
        )

        finite_coordinate_groups = []

        if background_coordinates is not None:
            finite_background = np.all(
                np.isfinite(background_coordinates),
                axis=1,
            )
            background_coordinates = background_coordinates[finite_background]

            if len(background_coordinates) > 0:
                ax.scatter(
                    background_coordinates[:, 0],
                    background_coordinates[:, 1],
                    background_coordinates[:, 2],
                    color=background_color,
                    marker="o",
                    s=background_size,
                    alpha=background_alpha,
                    linewidths=0.0,
                    label="All PMTs",
                )
                finite_coordinate_groups.append(background_coordinates)

        hit_collection = None

        if len(plot_values) > 0:
            hit_collection = ax.scatter(
                plot_coordinates[:, 0],
                plot_coordinates[:, 1],
                plot_coordinates[:, 2],
                c=plot_values,
                marker="o",
                s=marker_size,
                alpha=alpha,
                cmap=cmap,
                norm=norm,
                linewidths=0.0,
            )
            finite_coordinate_groups.append(plot_coordinates)

        if colorbar and hit_collection is not None:
            figure.colorbar(
                hit_collection,
                ax=ax,
                pad=0.08,
                shrink=0.75,
                label=color_label,
            )

        if finite_coordinate_groups:
            display_coordinates = np.concatenate(
                finite_coordinate_groups,
                axis=0,
            )
            _set_3d_limits(
                ax,
                display_coordinates,
                equal_aspect=equal_aspect,
            )

        if title is not None:
            ax.set_title(title)

        if show_axes:
            ax.set_xlabel(axis_labels[0])
            ax.set_ylabel(axis_labels[1])
            ax.set_zlabel(axis_labels[2])

            ax.grid(show_grid)

            if show_background_panes:
                ax.xaxis.set_pane_color(pane_color)
                ax.yaxis.set_pane_color(pane_color)
                ax.zaxis.set_pane_color(pane_color)
            else:
                transparent = (
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                )
                ax.xaxis.set_pane_color(transparent)
                ax.yaxis.set_pane_color(transparent)
                ax.zaxis.set_pane_color(transparent)
        else:
            ax.set_axis_off()

    return figure, ax


def _as_values(
    values: ArrayLike,
    *,
    name: str,
) -> NDArray[np.float64]:
    array = np.asarray(
        values,
        dtype=float,
    )

    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]

    if array.ndim != 1:
        raise ValueError(f"{name} must have shape (N,), got {array.shape}.")

    return array


def _as_coordinates(
    values: ArrayLike,
    *,
    dimensions: int,
    name: str,
) -> NDArray[np.float64]:
    array = np.asarray(
        values,
        dtype=float,
    )

    if array.ndim != 2 or array.shape[1] != dimensions:
        raise ValueError(
            f"{name} must have shape (N, {dimensions}), got {array.shape}."
        )

    return array


def _validate_same_length(
    first: np.ndarray,
    second: np.ndarray,
    *,
    first_name: str,
    second_name: str,
) -> None:
    if len(first) != len(second):
        raise ValueError(
            "Array lengths do not match: "
            f"{first_name}={len(first)}, "
            f"{second_name}={len(second)}"
        )


def _default_norm(
    values: NDArray[np.float64],
) -> colors.Normalize:
    if len(values) == 0:
        return colors.Normalize(
            vmin=0.0,
            vmax=1.0,
        )

    positive = values[values > 0.0]

    if len(positive) > 0:
        return colors.LogNorm(
            vmin=float(np.min(positive)),
            vmax=float(np.max(positive)),
        )

    minimum = float(np.min(values))
    maximum = float(np.max(values))

    if maximum <= minimum:
        maximum = minimum + 1.0

    return colors.Normalize(
        vmin=minimum,
        vmax=maximum,
    )


def _set_3d_limits(
    ax: Axes,
    coordinates: NDArray[np.float64],
    *,
    equal_aspect: bool,
) -> None:
    minimum = np.min(
        coordinates,
        axis=0,
    )
    maximum = np.max(
        coordinates,
        axis=0,
    )

    centers = 0.5 * (minimum + maximum)
    ranges = maximum - minimum

    if equal_aspect:
        half_range = 0.5 * float(np.max(ranges))

        if half_range == 0.0:
            half_range = 1.0

        ax.set_xlim(
            centers[0] - half_range,
            centers[0] + half_range,
        )
        ax.set_ylim(
            centers[1] - half_range,
            centers[1] + half_range,
        )
        ax.set_zlim(
            centers[2] - half_range,
            centers[2] + half_range,
        )
        ax.set_box_aspect((1.0, 1.0, 1.0))
    else:
        padding = np.where(
            ranges > 0.0,
            0.05 * ranges,
            1.0,
        )

        ax.set_xlim(
            minimum[0] - padding[0],
            maximum[0] + padding[0],
        )
        ax.set_ylim(
            minimum[1] - padding[1],
            maximum[1] + padding[1],
        )
        ax.set_zlim(
            minimum[2] - padding[2],
            maximum[2] + padding[2],
        )


def add_cylindrical_detector_3d(
    ax: Axes,
    *,
    radius: float,
    half_height: float,
    axis: int = 2,
    number_of_azimuth_lines: int = 16,
    number_of_axial_rings: int = 9,
    color: str = "0.45",
    linewidth: float = 0.7,
    alpha: float = 0.45,
) -> Axes:
    """Add a cylindrical detector wireframe to a 3D Axes."""
    if radius <= 0.0:
        raise ValueError("radius must be positive.")

    if half_height <= 0.0:
        raise ValueError("half_height must be positive.")

    if axis not in {0, 1, 2}:
        raise ValueError("axis must be 0, 1, or 2.")

    azimuth = np.linspace(
        -np.pi,
        np.pi,
        200,
    )
    axial_values = np.linspace(
        -half_height,
        half_height,
        number_of_axial_rings,
    )

    # Barrel rings
    for axial in axial_values:
        transverse_first = radius * np.cos(azimuth)
        transverse_second = radius * np.sin(azimuth)
        axial_coordinates = np.full_like(
            azimuth,
            axial,
        )

        coordinates = _cylindrical_components_to_xyz(
            transverse_first,
            transverse_second,
            axial_coordinates,
            axis=axis,
        )

        ax.plot(
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )

    # Lines parallel to the cylinder axis
    line_azimuths = np.linspace(
        -np.pi,
        np.pi,
        number_of_azimuth_lines,
        endpoint=False,
    )

    for angle in line_azimuths:
        axial_coordinates = np.array(
            [-half_height, half_height],
            dtype=float,
        )
        transverse_first = np.full(
            2,
            radius * np.cos(angle),
        )
        transverse_second = np.full(
            2,
            radius * np.sin(angle),
        )

        coordinates = _cylindrical_components_to_xyz(
            transverse_first,
            transverse_second,
            axial_coordinates,
            axis=axis,
        )

        ax.plot(
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )

    # Radial lines on the two endcaps
    for axial in (-half_height, half_height):
        for angle in line_azimuths:
            transverse_first = np.array([0.0, radius * np.cos(angle)])
            transverse_second = np.array([0.0, radius * np.sin(angle)])
            axial_coordinates = np.full(
                2,
                axial,
            )

            coordinates = _cylindrical_components_to_xyz(
                transverse_first,
                transverse_second,
                axial_coordinates,
                axis=axis,
            )

            ax.plot(
                coordinates[:, 0],
                coordinates[:, 1],
                coordinates[:, 2],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
            )

    return ax


def _cylindrical_components_to_xyz(
    transverse_first: ArrayLike,
    transverse_second: ArrayLike,
    axial: ArrayLike,
    *,
    axis: int,
) -> NDArray[np.float64]:
    """Arrange transverse and axial components as x, y, z."""
    transverse_first = np.asarray(
        transverse_first,
        dtype=float,
    )
    transverse_second = np.asarray(
        transverse_second,
        dtype=float,
    )
    axial = np.asarray(
        axial,
        dtype=float,
    )

    coordinates = np.empty(
        (len(axial), 3),
        dtype=float,
    )

    transverse_axes = [index for index in range(3) if index != axis]

    coordinates[:, transverse_axes[0]] = transverse_first
    coordinates[:, transverse_axes[1]] = transverse_second
    coordinates[:, axis] = axial

    return coordinates


def plot_cylindrical_event_2d(
    values: ArrayLike,
    hit_coordinates: ArrayLike,
    *,
    radius: float,
    half_height: float,
    all_pmt_coordinates: ArrayLike | None = None,
    axis: int = 2,
    title: str | None = None,
    color_label: str = "Value",
    cmap: str = "plasma",
    norm: colors.Normalize | None = None,
    show_zero: bool = False,
    surface_tolerance: float | None = None,
    background_color: str = "0.82",
    background_size: float = 5.0,
    marker_size: float = 18.0,
    figsize: tuple[float, float] = (18.0, 6.0),
) -> tuple[Figure, NDArray[np.object_]]:
    """Plot barrel and endcaps as an unfolded cylindrical display."""
    if axis not in {0, 1, 2}:
        raise ValueError("axis must be 0, 1, or 2.")

    if radius <= 0.0:
        raise ValueError("radius must be positive.")

    if half_height <= 0.0:
        raise ValueError("half_height must be positive.")

    values = _as_values(
        values,
        name="values",
    )
    hit_coordinates = _as_coordinates(
        hit_coordinates,
        dimensions=3,
        name="hit_coordinates",
    )

    _validate_same_length(
        values,
        hit_coordinates,
        first_name="values",
        second_name="hit_coordinates",
    )

    if surface_tolerance is None:
        surface_tolerance = 0.05 * min(
            radius,
            half_height,
        )

    selected = np.isfinite(values)
    selected &= np.all(
        np.isfinite(hit_coordinates),
        axis=1,
    )

    if not show_zero:
        selected &= values != 0.0

    values = values[selected]
    hit_coordinates = hit_coordinates[selected]

    if norm is None:
        norm = _default_norm(values)

    figure, axes = plt.subplots(
        1,
        3,
        figsize=figsize,
        gridspec_kw={
            "width_ratios": [1.0, 2.2, 1.0],
        },
    )

    bottom_axis, barrel_axis, top_axis = axes

    if all_pmt_coordinates is not None:
        all_pmt_coordinates = _as_coordinates(
            all_pmt_coordinates,
            dimensions=3,
            name="all_pmt_coordinates",
        )

        _draw_cylindrical_projection(
            axes,
            all_pmt_coordinates,
            None,
            radius=radius,
            half_height=half_height,
            axis=axis,
            tolerance=surface_tolerance,
            color=background_color,
            marker_size=background_size,
        )

    collection = _draw_cylindrical_projection(
        axes,
        hit_coordinates,
        values,
        radius=radius,
        half_height=half_height,
        axis=axis,
        tolerance=surface_tolerance,
        cmap=cmap,
        norm=norm,
        marker_size=marker_size,
    )

    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        300,
    )

    for endcap_axis in (bottom_axis, top_axis):
        endcap_axis.plot(
            radius * np.cos(theta),
            radius * np.sin(theta),
            color="black",
            linewidth=1.0,
        )
        endcap_axis.set_xlim(-radius, radius)
        endcap_axis.set_ylim(-radius, radius)
        endcap_axis.set_aspect(
            "equal",
            adjustable="box",
        )
        endcap_axis.grid(alpha=0.2)

    bottom_axis.set_title("Bottom endcap")
    top_axis.set_title("Top endcap")

    barrel_axis.set_title("Barrel")
    barrel_axis.set_xlabel(r"Azimuthal distance $R\phi$")
    barrel_axis.set_ylabel("Axial position")
    barrel_axis.set_xlim(
        -np.pi * radius,
        np.pi * radius,
    )
    barrel_axis.set_ylim(
        -half_height,
        half_height,
    )
    barrel_axis.grid(alpha=0.2)

    bottom_axis.set_xlabel("Transverse coordinate 1")
    bottom_axis.set_ylabel("Transverse coordinate 2")
    top_axis.set_xlabel("Transverse coordinate 1")
    top_axis.set_ylabel("Transverse coordinate 2")

    if title is not None:
        figure.suptitle(
            title,
            fontsize=15,
        )

    if collection is not None:
        figure.colorbar(
            collection,
            ax=axes.tolist(),
            pad=0.02,
            shrink=0.85,
            label=color_label,
        )

    figure.subplots_adjust(
        top=0.84,
        bottom=0.14,
        left=0.06,
        right=0.91,
        wspace=0.28,
    )

    return figure, axes


def _draw_cylindrical_projection(
    axes: NDArray[np.object_],
    coordinates: NDArray[np.float64],
    values: NDArray[np.float64] | None,
    *,
    radius: float,
    half_height: float,
    axis: int,
    tolerance: float,
    color: str | None = None,
    cmap: str = "plasma",
    norm: colors.Normalize | None = None,
    marker_size: float,
):
    bottom_axis, barrel_axis, top_axis = axes

    axial = coordinates[:, axis]

    transverse_axes = [index for index in range(3) if index != axis]

    transverse_first = coordinates[
        :,
        transverse_axes[0],
    ]
    transverse_second = coordinates[
        :,
        transverse_axes[1],
    ]

    transverse_radius = np.hypot(
        transverse_first,
        transverse_second,
    )
    azimuth = np.arctan2(
        transverse_second,
        transverse_first,
    )

    distance_to_barrel = np.abs(transverse_radius - radius)
    distance_to_endcap = np.abs(np.abs(axial) - half_height)

    barrel = distance_to_barrel <= distance_to_endcap
    top = ~barrel & (axial >= 0.0)
    bottom = ~barrel & (axial < 0.0)

    # Keep slightly off-surface PMTs if they are still recognizably
    # associated with the nearest detector surface.
    on_barrel = barrel & (distance_to_barrel <= tolerance)
    on_top = top & (distance_to_endcap <= tolerance)
    on_bottom = bottom & (distance_to_endcap <= tolerance)

    if not np.any(on_barrel | on_top | on_bottom):
        on_barrel = barrel
        on_top = top
        on_bottom = bottom

    scatter_kwargs: dict[str, Any] = {
        "s": marker_size,
        "linewidths": 0.0,
    }

    if values is None:
        scatter_kwargs["color"] = color
        scatter_kwargs["alpha"] = 0.45
    else:
        scatter_kwargs["cmap"] = cmap
        scatter_kwargs["norm"] = norm

    collection = None

    if np.any(on_barrel):
        collection = barrel_axis.scatter(
            radius * azimuth[on_barrel],
            axial[on_barrel],
            c=(values[on_barrel] if values is not None else None),
            **scatter_kwargs,
        )

    if np.any(on_bottom):
        bottom_axis.scatter(
            transverse_first[on_bottom],
            transverse_second[on_bottom],
            c=(values[on_bottom] if values is not None else None),
            **scatter_kwargs,
        )

    if np.any(on_top):
        top_collection = top_axis.scatter(
            transverse_first[on_top],
            transverse_second[on_top],
            c=(values[on_top] if values is not None else None),
            **scatter_kwargs,
        )

        if collection is None:
            collection = top_collection

    return collection


def plot_unrolled_cylindrical_event(
    values: ArrayLike,
    hit_coordinates: ArrayLike,
    *,
    radius: float,
    half_height: float,
    all_pmt_coordinates: ArrayLike | None = None,
    axis: int = 2,
    ax: Axes | None = None,
    title: str | None = None,
    color_label: str = "Value",
    cmap: str = "plasma",
    norm: colors.Normalize | None = None,
    show_zero: bool = False,
    endcap_gap: float | None = None,
    background_color: str = "0.82",
    background_size: float = 5.0,
    background_alpha: float = 0.45,
    marker_size: float = 18.0,
    alpha: float = 1.0,
    colorbar: bool = True,
    show_grid: bool = True,
    figsize: tuple[float, float] = (14.0, 9.0),
) -> tuple[Figure, Axes]:
    """Plot a cylindrical detector as a single unrolled 2D surface.

    The barrel is displayed as a rectangle using (R * azimuth, axial).
    The top and bottom endcaps are displayed as circles above and below
    the barrel.

    Parameters
    ----------
    values
        Values associated with hit PMTs.
    hit_coordinates
        Cartesian hit-PMT coordinates with shape (N, 3).
    radius
        Cylinder radius.
    half_height
        Cylinder half-height along the selected axis.
    all_pmt_coordinates
        Optional Cartesian coordinates of every PMT.
    axis
        Cylinder axis: 0, 1, or 2.
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive.")

    if half_height <= 0.0:
        raise ValueError("half_height must be positive.")

    if axis not in {0, 1, 2}:
        raise ValueError("axis must be 0, 1, or 2.")

    values_array = _as_values(
        values,
        name="values",
    )
    hit_coordinates_array = _as_coordinates(
        hit_coordinates,
        dimensions=3,
        name="hit_coordinates",
    )

    _validate_same_length(
        values_array,
        hit_coordinates_array,
        first_name="values",
        second_name="hit_coordinates",
    )

    finite = np.isfinite(values_array)
    finite &= np.all(
        np.isfinite(hit_coordinates_array),
        axis=1,
    )

    if not show_zero:
        finite &= values_array != 0.0

    plot_values = values_array[finite]
    plot_coordinates = hit_coordinates_array[finite]

    if norm is None:
        norm = _default_norm(plot_values)

    if endcap_gap is None:
        endcap_gap = 0.15 * radius

    if ax is None:
        figure, ax = plt.subplots(
            figsize=figsize,
        )
    else:
        figure = ax.figure

    if all_pmt_coordinates is not None:
        background_coordinates = _as_coordinates(
            all_pmt_coordinates,
            dimensions=3,
            name="all_pmt_coordinates",
        )

        finite_background = np.all(
            np.isfinite(background_coordinates),
            axis=1,
        )
        background_coordinates = background_coordinates[finite_background]

        background_xy = _unroll_cylindrical_coordinates(
            background_coordinates,
            radius=radius,
            half_height=half_height,
            axis=axis,
            endcap_gap=endcap_gap,
        )

        ax.scatter(
            background_xy[:, 0],
            background_xy[:, 1],
            color=background_color,
            s=background_size,
            alpha=background_alpha,
            linewidths=0.0,
            zorder=1,
        )

    hit_xy = _unroll_cylindrical_coordinates(
        plot_coordinates,
        radius=radius,
        half_height=half_height,
        axis=axis,
        endcap_gap=endcap_gap,
    )

    hit_collection = None

    if len(plot_values) > 0:
        hit_collection = ax.scatter(
            hit_xy[:, 0],
            hit_xy[:, 1],
            c=plot_values,
            s=marker_size,
            cmap=cmap,
            norm=norm,
            alpha=alpha,
            linewidths=0.0,
            zorder=3,
        )

    _draw_unrolled_cylinder_outline(
        ax,
        radius=radius,
        half_height=half_height,
        endcap_gap=endcap_gap,
    )

    if colorbar and hit_collection is not None:
        figure.colorbar(
            hit_collection,
            ax=ax,
            pad=0.02,
            label=color_label,
        )

    if title is not None:
        ax.set_title(title)

    azimuth_ticks = np.array(
        [
            -np.pi,
            -np.pi / 2.0,
            0.0,
            np.pi / 2.0,
            np.pi,
        ]
    )

    ax.set_xticks(
        radius * azimuth_ticks,
        labels=(
            r"$-\pi$",
            r"$-\pi/2$",
            "0",
            r"$\pi/2$",
            r"$\pi$",
        ),
    )
    ax.set_xlabel("Azimuth")

    # The vertical coordinate combines the barrel and endcaps, so its
    # numerical scale is intentionally hidden.
    ax.set_ylabel("")
    ax.set_yticks([])

    horizontal_limit = np.pi * radius

    bottom_center = -half_height - endcap_gap - radius
    top_center = half_height + endcap_gap + radius

    ax.set_xlim(
        -horizontal_limit,
        horizontal_limit,
    )
    ax.set_ylim(
        bottom_center - 1.1 * radius,
        top_center + 1.1 * radius,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    if show_grid:
        axial_grid = np.linspace(
            -half_height,
            half_height,
            7,
        )

        for axial_position in axial_grid:
            ax.plot(
                [
                    -np.pi * radius,
                    np.pi * radius,
                ],
                [
                    axial_position,
                    axial_position,
                ],
                color="0.75",
                linestyle=":",
                linewidth=0.7,
                alpha=0.5,
                zorder=0,
            )

        for azimuth in (
            -np.pi / 2.0,
            0.0,
            np.pi / 2.0,
        ):
            x_position = radius * azimuth

            ax.plot(
                [x_position, x_position],
                [-half_height, half_height],
                color="0.75",
                linestyle=":",
                linewidth=0.7,
                alpha=0.5,
                zorder=0,
            )

    return figure, ax


def _unroll_cylindrical_coordinates(
    coordinates: NDArray[np.float64],
    *,
    radius: float,
    half_height: float,
    axis: int,
    endcap_gap: float,
) -> NDArray[np.float64]:
    """Map Cartesian PMT coordinates to a cylindrical net."""
    axial = coordinates[:, axis]

    transverse_axes = [index for index in range(3) if index != axis]

    transverse_first = coordinates[
        :,
        transverse_axes[0],
    ]
    transverse_second = coordinates[
        :,
        transverse_axes[1],
    ]

    transverse_radius = np.hypot(
        transverse_first,
        transverse_second,
    )
    azimuth = np.arctan2(
        transverse_second,
        transverse_first,
    )

    distance_to_barrel = np.abs(transverse_radius - radius)
    distance_to_endcap = np.abs(np.abs(axial) - half_height)

    barrel = distance_to_barrel <= distance_to_endcap
    top = ~barrel & (axial >= 0.0)
    bottom = ~barrel & (axial < 0.0)

    projected = np.empty(
        (len(coordinates), 2),
        dtype=float,
    )

    # Barrel:
    # horizontal = arc length along circumference
    # vertical = position along cylinder axis
    projected[barrel, 0] = radius * azimuth[barrel]
    projected[barrel, 1] = axial[barrel]

    top_center = half_height + endcap_gap + radius

    projected[top, 0] = transverse_first[top]
    projected[top, 1] = transverse_second[top] + top_center

    bottom_center = -half_height - endcap_gap - radius

    # Reverse one transverse coordinate so that the bottom cap is
    # viewed consistently from outside the detector.
    projected[bottom, 0] = -transverse_first[bottom]
    projected[bottom, 1] = transverse_second[bottom] + bottom_center

    return projected


def _draw_unrolled_cylinder_outline(
    ax: Axes,
    *,
    radius: float,
    half_height: float,
    endcap_gap: float,
) -> None:
    """Draw barrel and endcap boundaries on an unfolded display."""
    barrel_left = -np.pi * radius
    barrel_right = np.pi * radius
    barrel_bottom = -half_height
    barrel_top = half_height

    rectangle_x = [
        barrel_left,
        barrel_right,
        barrel_right,
        barrel_left,
        barrel_left,
    ]
    rectangle_y = [
        barrel_bottom,
        barrel_bottom,
        barrel_top,
        barrel_top,
        barrel_bottom,
    ]

    ax.plot(
        rectangle_x,
        rectangle_y,
        color="black",
        linewidth=1.2,
        zorder=2,
    )

    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        400,
    )

    top_center = half_height + endcap_gap + radius
    bottom_center = -half_height - endcap_gap - radius

    for center in (
        bottom_center,
        top_center,
    ):
        ax.plot(
            radius * np.cos(theta),
            center + radius * np.sin(theta),
            color="black",
            linewidth=1.2,
            zorder=2,
        )

    # Barrel seam at azimuth = +/- pi.
    ax.axvline(
        -np.pi * radius,
        color="0.5",
        linestyle="--",
        linewidth=0.8,
        alpha=0.6,
    )
    ax.axvline(
        np.pi * radius,
        color="0.5",
        linestyle="--",
        linewidth=0.8,
        alpha=0.6,
    )
