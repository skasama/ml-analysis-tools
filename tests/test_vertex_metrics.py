import numpy as np
import pytest

from analysis_tools.analysis.vertex_regression import (
    analyze_vertex_regression,
    calculate_vertex_quantities,
)
from analysis_tools.metrics.vertex import (
    vertex_errors,
    vertex_residuals,
)


def test_vertex_residuals() -> None:
    predictions = np.array(
        [
            [1.0, 2.0, 3.0],
            [5.0, 5.0, 5.0],
        ]
    )
    targets = np.array(
        [
            [0.0, 1.0, 2.0],
            [4.0, 6.0, 5.0],
        ]
    )

    actual = vertex_residuals(predictions, targets)

    np.testing.assert_allclose(
        actual,
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, 0.0],
        ],
    )


def test_vertex_errors() -> None:
    predictions = np.array(
        [
            [3.0, 4.0, 0.0],
            [0.0, 0.0, 2.0],
        ]
    )
    targets = np.zeros((2, 3))

    actual = vertex_errors(predictions, targets)

    np.testing.assert_allclose(actual, [5.0, 2.0])


def test_vertex_metrics_accept_singleton_dimension() -> None:
    predictions = np.array([[[3.0, 4.0, 0.0]]])
    targets = np.array([[[0.0, 0.0, 0.0]]])

    actual = vertex_errors(predictions, targets)

    np.testing.assert_allclose(actual, [5.0])


def test_vertex_result_to_quantities() -> None:
    result = analyze_vertex_regression(
        predictions=[[1.0, 2.0, 3.0]],
        targets=[[0.0, 1.0, 2.0]],
        detector_radius=100.0,
        detector_half_height=100.0,
    )

    quantities = result.to_quantities()

    np.testing.assert_allclose(quantities.residual_x, [1.0])
    np.testing.assert_allclose(quantities.residual_y, [1.0])
    np.testing.assert_allclose(quantities.residual_z, [1.0])


def test_vertex_residuals_reject_length_mismatch() -> None:
    with pytest.raises(ValueError, match="Array lengths do not match"):
        vertex_residuals(
            predictions=np.zeros((2, 3)),
            targets=np.zeros((1, 3)),
        )


def test_vertex_residuals_reject_invalid_shape() -> None:
    with pytest.raises(ValueError, match="predictions"):
        vertex_residuals(
            predictions=np.zeros((2, 2)),
            targets=np.zeros((2, 3)),
        )


def test_calculate_vertex_quantities() -> None:
    predictions = np.array(
        [
            [3.0, 4.0, 2.0],
            [0.0, 2.0, 5.0],
        ]
    )

    targets = np.array(
        [
            [0.0, 0.0, 2.0],
            [0.0, 1.0, 1.0],
        ]
    )

    result = calculate_vertex_quantities(
        predictions,
        targets,
    )

    np.testing.assert_allclose(
        result.residuals,
        [
            [3.0, 4.0, 0.0],
            [0.0, 1.0, 4.0],
        ],
    )

    np.testing.assert_allclose(
        result.prediction_r,
        [5.0, 2.0],
    )

    np.testing.assert_allclose(
        result.target_r,
        [0.0, 1.0],
    )

    np.testing.assert_allclose(
        result.residual_r,
        [5.0, 1.0],
    )

    np.testing.assert_allclose(
        result.errors_3d,
        [
            5.0,
            np.sqrt(17.0),
        ],
    )

    assert result.event_count == 2


def test_vertex_quantities_select() -> None:
    quantities = calculate_vertex_quantities(
        predictions=np.array(
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        ),
        targets=np.zeros((3, 3)),
    )

    selected = quantities.select(np.array([True, False, True]))

    assert selected.event_count == 2

    np.testing.assert_allclose(
        selected.residual_x,
        [1.0, 3.0],
    )
