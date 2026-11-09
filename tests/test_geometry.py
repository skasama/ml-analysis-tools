import numpy as np
import pytest

from analysis.geometry import opening_angle, towall, unit, wall


def test_unit_normalizes_vectors():
    result = unit([[3, 4, 0], [0, 0, -2]])

    np.testing.assert_allclose(result, [[0.6, 0.8, 0], [0, 0, -1]])


def test_unit_rejects_zero_length_vectors():
    with pytest.raises(ValueError, match="Zero-length"):
        unit([[0, 0, 0]])


def test_opening_angle_is_in_degrees():
    result = opening_angle([[1, 0, 0], [1, 0, 0]], [[1, 0, 0], [0, 1, 0]])

    np.testing.assert_allclose(result, [0, 90])


def test_wall_returns_nearest_boundary_distance():
    result = wall([[0, 0, 0], [100, 0, 0], [0, 0, 100]])

    np.testing.assert_allclose(result, [1690, 1590, 1690])


def test_towall_finds_forward_cylinder_or_cap_intersection():
    positions = np.array([[0, 0, 0], [0, 0, 1700]])
    directions = np.array([[1, 0, 0], [0, 0, 1]])

    np.testing.assert_allclose(towall(positions, directions), [1690, 110])
