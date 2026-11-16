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


def test_compute_towall_radial_direction() -> None:
    actual = compute_towall(
        positions=[[0.0, 0.0, 0.0]],
        directions=[[1.0, 0.0, 0.0]],
        radius=10.0,
        half_height=20.0,
    )

    np.testing.assert_allclose(actual, [10.0])


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
