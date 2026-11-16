import numpy as np
import pytest

from analysis_tools.data.arrays import (
    as_scalar_array,
    as_vector_array,
    validate_same_length,
)


def test_as_scalar_array_keeps_one_dimensional_array() -> None:
    values = np.array([1.0, 2.0, 3.0])

    actual = as_scalar_array(values)

    np.testing.assert_array_equal(actual, values)


def test_as_scalar_array_removes_singleton_dimensions() -> None:
    values = np.array([[[1.0]], [[2.0]], [[3.0]]])

    actual = as_scalar_array(values)

    assert actual.shape == (3,)
    np.testing.assert_array_equal(actual, [1.0, 2.0, 3.0])


def test_as_scalar_array_rejects_multiple_values_per_event() -> None:
    values = np.ones((3, 2))

    with pytest.raises(ValueError, match="must have shape"):
        as_scalar_array(values, name="energy")


def test_as_vector_array_keeps_standard_shape() -> None:
    values = np.arange(9).reshape(3, 3)

    actual = as_vector_array(values)

    np.testing.assert_array_equal(actual, values)


def test_as_vector_array_removes_singleton_dimensions() -> None:
    values = np.arange(9).reshape(3, 1, 3)

    actual = as_vector_array(values)

    assert actual.shape == (3, 3)


def test_as_vector_array_rejects_invalid_shape() -> None:
    values = np.ones((3, 2))

    with pytest.raises(ValueError, match="direction"):
        as_vector_array(values, name="direction")


def test_validate_same_length() -> None:
    length = validate_same_length(
        np.zeros((3, 3)),
        np.zeros(3),
        names=("positions", "energies"),
    )

    assert length == 3


def test_validate_same_length_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match="predictions=3, targets=2"):
        validate_same_length(
            np.zeros(3),
            np.zeros(2),
            names=("predictions", "targets"),
        )
