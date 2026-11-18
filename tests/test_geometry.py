import numpy as np
import pytest

from analysis_tools.transforms.geometry import (
    compute_towall,
    compute_wall,
    cylindrical_radius,
    normalize_vectors,
)


def test_normalize_vectors() -> None:
    vectors = np.array(
        [
            [3.0, 0.0, 0.0],
            [0.0, 0.0, -2.0],
        ]
    )

    actual = normalize_vectors(vectors)

    np.testing.assert_allclose(
        actual,
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
    )


def test_normalize_vectors_rejects_zero_vector() -> None:
    with pytest.raises(ValueError, match="zero-length"):
        normalize_vectors([[0.0, 0.0, 0.0]])


def test_cylindrical_radius() -> None:
    positions = np.array(
        [
            [3.0, 4.0, 10.0],
            [5.0, 12.0, -20.0],
        ]
    )

    actual = cylindrical_radius(positions)

    np.testing.assert_allclose(actual, [5.0, 13.0])


def test_compute_wall_at_center() -> None:
    actual = compute_wall(
        [[0.0, 0.0, 0.0]],
        radius=10.0,
        half_height=20.0,
    )

    np.testing.assert_allclose(actual, [10.0])


def test_compute_wall_uses_nearest_boundary() -> None:
    positions = np.array(
        [
            [9.0, 0.0, 0.0],
            [0.0, 0.0, 18.0],
        ]
    )

    actual = compute_wall(
        positions,
        radius=10.0,
        half_height=20.0,
    )

    np.testing.assert_allclose(actual, [1.0, 2.0])


def test_compute_wall_is_negative_outside() -> None:
    actual = compute_wall(
        [[11.0, 0.0, 0.0]],
        radius=10.0,
        half_height=20.0,
    )

    np.testing.assert_allclose(actual, [-1.0])


def test_compute_towall_vertical_direction() -> None:
    actual = compute_towall(
        positions=[[0.0, 0.0, 5.0]],
        directions=[[0.0, 0.0, 1.0]],
        radius=10.0,
        half_height=20.0,
    )

    np.testing.assert_allclose(actual, [15.0])


def test_compute_towall_normalizes_direction() -> None:
    actual = compute_towall(
        positions=[[0.0, 0.0, 0.0]],
        directions=[[2.0, 0.0, 0.0]],
        radius=10.0,
        half_height=20.0,
    )

    np.testing.assert_allclose(actual, [10.0])


def test_compute_towall_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_towall(
            positions=[[0.0, 0.0, 0.0]],
            directions=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            radius=10.0,
            half_height=20.0,
        )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_cylindrical_radius_for_each_axis(
    axis: int,
) -> None:
    position = np.zeros((1, 3))
    transverse_axes = [index for index in range(3) if index != axis]

    position[0, transverse_axes[0]] = 3.0
    position[0, transverse_axes[1]] = 4.0
    position[0, axis] = 10.0

    actual = cylindrical_radius(
        position,
        axis=axis,
    )

    np.testing.assert_allclose(actual, [5.0])


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_compute_wall_for_each_axis(
    axis: int,
) -> None:
    position = np.zeros((1, 3))
    position[0, axis] = 18.0

    actual = compute_wall(
        position,
        radius=10.0,
        half_height=20.0,
        axis=axis,
    )

    np.testing.assert_allclose(actual, [2.0])


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_compute_towall_axial_direction(
    axis: int,
) -> None:
    position = np.zeros((1, 3))
    position[0, axis] = 5.0

    direction = np.zeros((1, 3))
    direction[0, axis] = 1.0

    actual = compute_towall(
        position,
        direction,
        radius=10.0,
        half_height=20.0,
        axis=axis,
    )

    np.testing.assert_allclose(actual, [15.0])


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_compute_towall_radial_direction(
    axis: int,
) -> None:
    transverse_axis = next(index for index in range(3) if index != axis)

    position = np.zeros((1, 3))
    direction = np.zeros((1, 3))
    direction[0, transverse_axis] = 1.0

    actual = compute_towall(
        position,
        direction,
        radius=10.0,
        half_height=20.0,
        axis=axis,
    )

    np.testing.assert_allclose(actual, [10.0])


def test_geometry_rejects_invalid_axis() -> None:
    with pytest.raises(
        ValueError,
        match="0, 1, or 2",
    ):
        compute_wall(
            [[0.0, 0.0, 0.0]],
            radius=10.0,
            half_height=20.0,
            axis=3,
        )


def test_compute_towall_rejects_outside_position() -> None:
    with pytest.raises(
        ValueError,
        match="outside",
    ):
        compute_towall(
            positions=[[11.0, 0.0, 0.0]],
            directions=[[1.0, 0.0, 0.0]],
            radius=10.0,
            half_height=20.0,
            axis=2,
        )
