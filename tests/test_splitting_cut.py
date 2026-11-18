import numpy as np
import pytest

from analysis_tools.physics import (
    Particle,
    momentum_to_energy,
)
from analysis_tools.splitting import (
    combine_cuts,
    energy_cut,
    hit_count_cut,
    keep_event_cut,
    label_cut,
    momentum_cut,
    range_cut,
    towall_cut,
    valid_data_cut,
    wall_cut,
)


def test_valid_data_cut() -> None:
    mask = valid_data_cut(
        [1.0, np.nan, 3.0],
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [np.inf, 0.0, 0.0],
        ],
    )

    np.testing.assert_array_equal(
        mask,
        [True, False, False],
    )


def test_label_cut() -> None:
    mask = label_cut(
        [0, 1, 2, 1],
        allowed_labels=[0, 2],
    )

    np.testing.assert_array_equal(
        mask,
        [True, False, True, False],
    )


def test_hit_count_cut_is_inclusive() -> None:
    mask = hit_count_cut(
        [99, 100, 200, 201],
        minimum=100,
        maximum=200,
    )

    np.testing.assert_array_equal(
        mask,
        [False, True, True, False],
    )


def test_keep_event_cut() -> None:
    mask = keep_event_cut(
        [1, 0, 1],
    )

    np.testing.assert_array_equal(
        mask,
        [True, False, True],
    )


def test_energy_cut() -> None:
    mask = energy_cut(
        [50.0, 100.0, 200.0],
        minimum=100.0,
    )

    np.testing.assert_array_equal(
        mask,
        [False, True, True],
    )


def test_momentum_cut_from_total_energy() -> None:
    labels = np.array(
        [
            Particle.MUON,
            Particle.ELECTRON,
            Particle.CHARGED_PION,
        ]
    )
    momentum = np.array([50.0, 100.0, 150.0])
    energies = momentum_to_energy(
        momentum,
        labels,
    )

    mask = momentum_cut(
        energies,
        labels,
        energy_type="total",
        minimum=100.0,
    )

    np.testing.assert_array_equal(
        mask,
        [False, True, True],
    )


def test_wall_cut() -> None:
    mask = wall_cut(
        [
            [0.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],
        ],
        detector_radius=10.0,
        detector_half_height=20.0,
        minimum=5.0,
    )

    np.testing.assert_array_equal(
        mask,
        [True, False],
    )


def test_towall_cut() -> None:
    mask = towall_cut(
        positions=[
            [0.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],
        ],
        directions=[
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        detector_radius=10.0,
        detector_half_height=20.0,
        minimum=5.0,
    )

    np.testing.assert_array_equal(
        mask,
        [True, False],
    )


def test_combine_cuts() -> None:
    mask = combine_cuts(
        [True, True, False],
        [True, False, True],
        [True, True, True],
    )

    np.testing.assert_array_equal(
        mask,
        [True, False, False],
    )


def test_range_cut_rejects_invalid_range() -> None:
    with pytest.raises(
        ValueError,
        match="maximum",
    ):
        range_cut(
            [1.0, 2.0],
            minimum=10.0,
            maximum=5.0,
        )


def test_combine_cuts_rejects_length_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="Array lengths do not match",
    ):
        combine_cuts(
            [True, False],
            [True],
        )


def test_wall_cut_with_y_axis() -> None:
    mask = wall_cut(
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, 19.0, 0.0],
        ],
        detector_radius=10.0,
        detector_half_height=20.0,
        axis=1,
        minimum=5.0,
    )

    np.testing.assert_array_equal(
        mask,
        [True, False],
    )


def test_towall_cut_with_y_axis() -> None:
    mask = towall_cut(
        positions=[
            [0.0, 5.0, 0.0],
            [0.0, 19.0, 0.0],
        ],
        directions=[
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        detector_radius=10.0,
        detector_half_height=20.0,
        axis=1,
        minimum=5.0,
    )

    np.testing.assert_array_equal(
        mask,
        [True, False],
    )
