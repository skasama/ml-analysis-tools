import numpy as np
import pytest

from analysis_tools.transforms.normalization import MinMaxTransform


def test_scalar_forward_transform() -> None:
    transform = MinMaxTransform(
        minimum=0.0,
        maximum=10.0,
    )

    actual = transform.forward([0.0, 5.0, 10.0])

    np.testing.assert_allclose(actual, [0.0, 0.5, 1.0])


def test_scalar_inverse_transform() -> None:
    transform = MinMaxTransform(
        minimum=0.0,
        maximum=10.0,
    )

    actual = transform.inverse([0.0, 0.5, 1.0])

    np.testing.assert_allclose(actual, [0.0, 5.0, 10.0])


def test_vector_transform() -> None:
    transform = MinMaxTransform(
        minimum=[-10.0, -20.0, -30.0],
        maximum=[10.0, 20.0, 30.0],
    )

    values = np.array(
        [
            [-10.0, -20.0, -30.0],
            [0.0, 0.0, 0.0],
            [10.0, 20.0, 30.0],
        ]
    )

    actual = transform.forward(values)

    np.testing.assert_allclose(
        actual,
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
        ],
    )


def test_forward_inverse_round_trip() -> None:
    transform = MinMaxTransform(
        minimum=[-1690.0, -1690.0, -1810.0],
        maximum=[1690.0, 1690.0, 1810.0],
    )
    expected = np.array(
        [
            [100.0, -200.0, 300.0],
            [-400.0, 500.0, -600.0],
        ]
    )

    actual = transform.inverse(transform.forward(expected))

    np.testing.assert_allclose(actual, expected)


def test_custom_normalized_range() -> None:
    transform = MinMaxTransform(
        minimum=-1.0,
        maximum=1.0,
        normalized_minimum=-1.0,
        normalized_maximum=1.0,
    )

    actual = transform.forward([-1.0, 0.0, 1.0])

    np.testing.assert_allclose(actual, [-1.0, 0.0, 1.0])


def test_rejects_invalid_physical_range() -> None:
    with pytest.raises(ValueError, match="greater than minimum"):
        MinMaxTransform(
            minimum=10.0,
            maximum=10.0,
        )


def test_rejects_different_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        MinMaxTransform(
            minimum=[0.0, 0.0],
            maximum=[1.0, 1.0, 1.0],
        )
