import matplotlib.pyplot as plt
import numpy as np
import pytest

from analysis_tools.plotting.event_display import (
    plot_event_2d,
    plot_event_3d,
)


def test_plot_event_2d() -> None:
    values = np.array([0.0, 1.0, 10.0])
    coordinates = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    original = values.copy()

    figure, axis = plot_event_2d(
        values,
        coordinates,
        show_zero=False,
        colorbar=False,
    )

    assert axis.figure is figure
    assert len(axis.collections) == 1
    np.testing.assert_array_equal(
        values,
        original,
    )

    plt.close(figure)


def test_plot_event_2d_with_background() -> None:
    figure, axis = plot_event_2d(
        values=[1.0, 2.0],
        hit_coordinates=[
            [0.0, 0.0],
            [1.0, 1.0],
        ],
        all_pmt_coordinates=[
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        colorbar=False,
        show_axes=True,
    )

    assert len(axis.collections) == 2
    assert axis.get_xlabel() == "x"
    assert axis.get_ylabel() == "y"

    plt.close(figure)


def test_plot_event_3d_has_xyz_axes() -> None:
    figure, axis = plot_event_3d(
        values=[1.0, 2.0],
        hit_coordinates=[
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
        ],
        colorbar=False,
    )

    assert axis.name == "3d"
    assert axis.get_xlabel() == "x"
    assert axis.get_ylabel() == "y"
    assert axis.get_zlabel() == "z"

    plt.close(figure)


def test_plot_event_3d_uses_equal_limits() -> None:
    figure, axis = plot_event_3d(
        values=[1.0, 2.0],
        hit_coordinates=[
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 4.0],
        ],
        colorbar=False,
        equal_aspect=True,
    )

    x_range = np.ptp(axis.get_xlim())
    y_range = np.ptp(axis.get_ylim())
    z_range = np.ptp(axis.get_zlim())

    assert x_range == pytest.approx(y_range)
    assert y_range == pytest.approx(z_range)

    plt.close(figure)


def test_event_display_rejects_length_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="Array lengths do not match",
    ):
        plot_event_3d(
            values=[1.0],
            hit_coordinates=[
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
        )


def test_plot_event_3d_rejects_standard_axes() -> None:
    figure, axis = plt.subplots()

    with pytest.raises(
        TypeError,
        match="three-dimensional",
    ):
        plot_event_3d(
            values=[1.0],
            hit_coordinates=[
                [0.0, 0.0, 0.0],
            ],
            ax=axis,
        )

    plt.close(figure)
