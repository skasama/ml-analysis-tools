import numpy as np
import pytest

from analysis.regression import inverse_minmax, scalar, xyz


def test_scalar_accepts_vector_and_single_column():
    values = np.array([1, 2])

    np.testing.assert_array_equal(scalar(values), values)
    np.testing.assert_array_equal(scalar(values[:, None]), values)


def test_scalar_rejects_other_shapes():
    with pytest.raises(ValueError, match="shape"):
        scalar(np.zeros((2, 2)))


def test_xyz_accepts_flattened_singleton_dimension():
    values = np.array([[1, 2, 3]])

    np.testing.assert_array_equal(xyz(values), values)
    np.testing.assert_array_equal(xyz(values[:, None, :]), values)


def test_xyz_rejects_other_shapes():
    with pytest.raises(ValueError, match="shape"):
        xyz(np.zeros((2, 2)))


def test_inverse_minmax_scales_values():
    np.testing.assert_allclose(inverse_minmax([0, 0.5, 1], -2, 2), [-2, 0, 2])


def test_inverse_minmax_accepts_minus_one_to_one_source_range():
    np.testing.assert_allclose(
        inverse_minmax([-1, 0, 1], 0, 10, source_min=-1, source_max=1),
        [0, 5, 10],
    )
