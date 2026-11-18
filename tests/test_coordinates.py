import numpy as np
import pytest

from analysis_tools.transforms.coordinates import (
    angles_from_direction,
    cartesian_to_cylindrical,
    cylindrical_to_cartesian,
    direction_from_angles,
)


def test_direction_from_angles_z_axis() -> None:
    actual = direction_from_angles(
        [
            [0.0, 0.0],
            [np.pi / 2.0, 0.0],
            [np.pi / 2.0, np.pi / 2.0],
        ],
        axis=2,
    )

    np.testing.assert_allclose(
        actual,
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        atol=1e-15,
    )


def test_direction_from_angles_y_axis() -> None:
    actual = direction_from_angles(
        [
            [0.0, 0.0],
            [np.pi / 2.0, 0.0],
        ],
        axis=1,
    )

    np.testing.assert_allclose(
        actual,
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        atol=1e-15,
    )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_direction_angle_round_trip(
    axis: int,
) -> None:
    expected = np.array(
        [
            [0.3, -1.0],
            [1.2, 0.5],
            [2.4, 2.0],
        ]
    )

    directions = direction_from_angles(
        expected,
        axis=axis,
    )
    actual = angles_from_direction(
        directions,
        axis=axis,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        atol=1e-14,
    )


def test_angles_from_non_unit_directions() -> None:
    actual = angles_from_direction(
        [
            [0.0, 0.0, 2.0],
            [3.0, 0.0, 0.0],
        ],
        axis=2,
    )

    np.testing.assert_allclose(
        actual,
        [
            [0.0, 0.0],
            [np.pi / 2.0, 0.0],
        ],
        atol=1e-15,
    )


def test_cartesian_to_cylindrical_z_axis() -> None:
    actual = cartesian_to_cylindrical(
        [
            [3.0, 4.0, 10.0],
            [0.0, -2.0, -5.0],
        ],
        axis=2,
    )

    np.testing.assert_allclose(
        actual,
        [
            [5.0, np.arctan2(4.0, 3.0), 10.0],
            [2.0, -np.pi / 2.0, -5.0],
        ],
    )


def test_cartesian_to_cylindrical_y_axis() -> None:
    actual = cartesian_to_cylindrical(
        [[3.0, 10.0, 4.0]],
        axis=1,
    )

    np.testing.assert_allclose(
        actual,
        [[5.0, np.arctan2(4.0, 3.0), 10.0]],
    )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_cylindrical_coordinate_round_trip(
    axis: int,
) -> None:
    expected = np.array(
        [
            [3.0, 4.0, 5.0],
            [-2.0, 6.0, -8.0],
            [1.0, -3.0, 7.0],
        ]
    )

    cylindrical = cartesian_to_cylindrical(
        expected,
        axis=axis,
    )
    actual = cylindrical_to_cartesian(
        cylindrical,
        axis=axis,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        atol=1e-14,
    )


def test_cylindrical_to_cartesian_rejects_negative_radius() -> None:
    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        cylindrical_to_cartesian(
            [[-1.0, 0.0, 0.0]],
        )


def test_coordinate_transform_rejects_invalid_axis() -> None:
    with pytest.raises(
        ValueError,
        match="0, 1, or 2",
    ):
        cartesian_to_cylindrical(
            [[1.0, 2.0, 3.0]],
            axis=3,
        )
