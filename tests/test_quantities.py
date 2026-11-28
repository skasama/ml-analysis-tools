import numpy as np
import pytest

from analysis_tools.quantities import (
    calculate_classification_quantities,
    calculate_direction_quantities,
    calculate_energy_quantities,
    calculate_vertex_quantities,
)


def test_calculate_energy_quantities() -> None:
    quantities = calculate_energy_quantities(
        predictions=[[110.0], [180.0]],
        targets=[100.0, 200.0],
    )

    np.testing.assert_allclose(quantities.residuals, [10.0, -20.0])
    np.testing.assert_allclose(
        quantities.relative_residuals,
        [0.1, -0.1],
    )
    assert quantities.event_count == 2


def test_energy_quantities_select() -> None:
    quantities = calculate_energy_quantities(
        predictions=[110.0, 180.0],
        targets=[100.0, 200.0],
    )

    selected = quantities.select([True, False])

    np.testing.assert_allclose(selected.predictions, [110.0])
    np.testing.assert_allclose(selected.targets, [100.0])


def test_calculate_direction_quantities() -> None:
    quantities = calculate_direction_quantities(
        predictions=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        targets=[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    )

    np.testing.assert_allclose(
        quantities.angular_errors,
        [0.0, 90.0],
    )
    assert quantities.prediction_theta.shape == (2,)
    assert quantities.prediction_phi.shape == (2,)
    assert quantities.target_theta.shape == (2,)
    assert quantities.target_phi.shape == (2,)
    np.testing.assert_allclose(quantities.theta_residuals, [0.0, 0.0])
    np.testing.assert_allclose(
        quantities.phi_residuals,
        [0.0, 90.0],
    )


def test_direction_quantities_wrap_phi_residual_to_shortest_angle() -> None:
    quantities = calculate_direction_quantities(
        predictions=[[np.cos(-np.pi + 0.1), np.sin(-np.pi + 0.1), 0.0]],
        targets=[[np.cos(np.pi - 0.1), np.sin(np.pi - 0.1), 0.0]],
    )

    np.testing.assert_allclose(quantities.phi_residuals, [np.degrees(0.2)])
    np.testing.assert_allclose(
        quantities.opening_angles,
        [np.degrees(0.2)],
    )
    np.testing.assert_allclose(
        quantities.prediction_phi,
        [np.degrees(-np.pi + 0.1)],
    )
    np.testing.assert_array_equal(
        quantities.angular_errors,
        quantities.opening_angles,
    )


def test_direction_quantities_opening_angle_in_radians() -> None:
    quantities = calculate_direction_quantities(
        predictions=[[0.0, 1.0, 0.0]],
        targets=[[1.0, 0.0, 0.0]],
        degrees=False,
    )

    np.testing.assert_allclose(quantities.opening_angles, [np.pi / 2.0])
    np.testing.assert_allclose(quantities.prediction_phi, [np.pi / 2.0])
    np.testing.assert_allclose(quantities.phi_residuals, [np.pi / 2.0])


def test_direction_quantities_select_preserves_derived_angles() -> None:
    quantities = calculate_direction_quantities(
        predictions=[
            [1.0, 0.0, 0.0],
            [np.cos(-np.pi + 0.1), np.sin(-np.pi + 0.1), 0.0],
        ],
        targets=[
            [1.0, 0.0, 0.0],
            [np.cos(np.pi - 0.1), np.sin(np.pi - 0.1), 0.0],
        ],
    )

    selected = quantities.select([False, True])

    assert selected.event_count == 1
    np.testing.assert_allclose(
        selected.phi_residuals,
        quantities.phi_residuals[1:],
    )
    np.testing.assert_allclose(
        selected.opening_angles,
        quantities.opening_angles[1:],
    )


def test_calculate_classification_quantities() -> None:
    quantities = calculate_classification_quantities(
        scores=[[2.0, 1.0], [1.0, 2.0]],
        targets=[0, 1],
    )

    np.testing.assert_array_equal(
        quantities.predicted_labels,
        [0, 1],
    )
    np.testing.assert_array_equal(
        quantities.correct_mask,
        [True, True],
    )
    np.testing.assert_allclose(
        quantities.probabilities.sum(axis=1),
        [1.0, 1.0],
    )


def test_classification_quantities_support_physical_class_labels() -> None:
    quantities = calculate_classification_quantities(
        scores=[[0.1, 0.9], [0.8, 0.2]],
        targets=[13, 13],
        class_labels=[13, 11],
    )

    np.testing.assert_array_equal(
        quantities.predicted_labels,
        [11, 13],
    )
    np.testing.assert_array_equal(
        quantities.correct_mask,
        [False, True],
    )


def test_classification_quantities_accept_explicit_probabilities() -> None:
    probabilities = np.array([[0.7, 0.3], [0.2, 0.8]])
    quantities = calculate_classification_quantities(
        scores=[[5.0, 1.0], [1.0, 5.0]],
        probabilities=probabilities,
        targets=[0, 1],
    )

    np.testing.assert_array_equal(quantities.probabilities, probabilities)


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        ([[0.7, 0.4]], "sum to 1"),
        ([[0.7, -0.1]], "range"),
        ([[0.7]], "same shape"),
        ([[np.nan, 1.0]], "finite"),
    ],
)
def test_classification_quantities_validate_explicit_probabilities(
    probabilities,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        calculate_classification_quantities(
            scores=[[1.0, 0.0]],
            probabilities=probabilities,
            targets=[0],
        )


@pytest.mark.parametrize(
    ("class_labels", "targets", "message"),
    [
        ([1], [1], "one label per score column"),
        ([1, 1], [1], "must be unique"),
        ([1, 2], [3], "missing from class_labels"),
    ],
)
def test_classification_quantities_validate_class_labels(
    class_labels,
    targets,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        calculate_classification_quantities(
            scores=[[1.0, 0.0]],
            targets=targets,
            class_labels=class_labels,
        )


def test_classification_quantities_select() -> None:
    quantities = calculate_classification_quantities(
        scores=[[1.0, 0.0], [0.0, 1.0]],
        targets=[0, 1],
    )

    selected = quantities.select([False, True])

    np.testing.assert_array_equal(selected.target_labels, [1])
    np.testing.assert_array_equal(selected.correct_mask, [True])


def test_calculate_vertex_quantities_keeps_event_level_arrays() -> None:
    quantities = calculate_vertex_quantities(
        predictions=[[1.0, 0.0, 3.0], [0.0, 2.0, 1.0]],
        targets=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )

    np.testing.assert_allclose(quantities.residual_x, [1.0, 0.0])
    np.testing.assert_allclose(quantities.residual_y, [0.0, 2.0])
    np.testing.assert_allclose(quantities.residual_z, [3.0, 1.0])
    np.testing.assert_allclose(quantities.residual_r, [1.0, 2.0])
    np.testing.assert_allclose(quantities.errors_3d, [np.sqrt(10.0), np.sqrt(5.0)])


def test_quantity_calculators_reject_length_mismatch() -> None:
    with pytest.raises(ValueError, match="Array lengths do not match"):
        calculate_energy_quantities(
            predictions=[100.0, 200.0],
            targets=[100.0],
        )
