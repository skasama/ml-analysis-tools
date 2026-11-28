import numpy as np
import pytest

from analysis_tools.metrics.direction import (
    direction_cosines,
    direction_errors,
    wrapped_angle_residuals,
)


def test_direction_cosines() -> None:
    predictions = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    targets = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    actual = direction_cosines(predictions, targets)

    np.testing.assert_allclose(actual, [1.0, 0.0])


def test_direction_errors_in_degrees() -> None:
    predictions = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    targets = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    actual = direction_errors(predictions, targets)

    np.testing.assert_allclose(actual, [0.0, 90.0, 180.0])


def test_direction_errors_in_radians() -> None:
    actual = direction_errors(
        predictions=[[0.0, 1.0, 0.0]],
        targets=[[1.0, 0.0, 0.0]],
        degrees=False,
    )

    np.testing.assert_allclose(actual, [np.pi / 2.0])


def test_direction_errors_normalize_input_vectors() -> None:
    actual = direction_errors(
        predictions=[[2.0, 0.0, 0.0]],
        targets=[[5.0, 0.0, 0.0]],
    )

    np.testing.assert_allclose(actual, [0.0])


def test_direction_errors_reject_length_mismatch() -> None:
    with pytest.raises(ValueError, match="Array lengths do not match"):
        direction_errors(
            predictions=np.ones((2, 3)),
            targets=np.ones((1, 3)),
        )


def test_direction_errors_reject_zero_vector() -> None:
    with pytest.raises(ValueError, match="zero-length"):
        direction_errors(
            predictions=[[0.0, 0.0, 0.0]],
            targets=[[1.0, 0.0, 0.0]],
        )


def test_wrapped_angle_residuals_use_shortest_signed_difference() -> None:
    actual = wrapped_angle_residuals(
        predictions=[-np.pi + 0.1, np.pi - 0.1],
        targets=[np.pi - 0.1, -np.pi + 0.1],
    )

    np.testing.assert_allclose(actual, [0.2, -0.2])


def test_wrapped_angle_residuals_rejects_non_one_dimensional_inputs() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        wrapped_angle_residuals(
            predictions=[[0.0]],
            targets=[[0.0]],
        )
