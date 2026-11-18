import numpy as np
import pytest

from analysis_tools.physics import (
    PARTICLE_MASS_MEV,
    Particle,
    energy_to_momentum,
    kinetic_energy_to_momentum,
    momentum_to_energy,
    momentum_to_kinetic_energy,
    particle_mass,
)


def test_particle_mass_from_scalar_label() -> None:
    actual = particle_mass(Particle.MUON)

    assert actual == pytest.approx(PARTICLE_MASS_MEV[Particle.MUON])


def test_particle_mass_from_label_array() -> None:
    actual = particle_mass([0, 1, 2])

    np.testing.assert_allclose(
        actual,
        [
            PARTICLE_MASS_MEV[Particle.MUON],
            PARTICLE_MASS_MEV[Particle.ELECTRON],
            PARTICLE_MASS_MEV[Particle.CHARGED_PION],
        ],
    )


def test_energy_to_momentum_at_rest() -> None:
    mass = PARTICLE_MASS_MEV[Particle.MUON]

    actual = energy_to_momentum(
        [mass],
        Particle.MUON,
    )

    np.testing.assert_allclose(actual, [0.0])


def test_energy_to_momentum_with_event_labels() -> None:
    labels = np.array([0, 1, 2])
    masses = particle_mass(labels)

    actual = energy_to_momentum(
        masses,
        labels,
    )

    np.testing.assert_allclose(actual, [0.0, 0.0, 0.0])


def test_energy_momentum_round_trip() -> None:
    labels = np.array([0, 1, 2])
    expected_momentum = np.array([100.0, 200.0, 300.0])

    energy = momentum_to_energy(
        expected_momentum,
        labels,
    )
    actual_momentum = energy_to_momentum(
        energy,
        labels,
    )

    np.testing.assert_allclose(
        actual_momentum,
        expected_momentum,
    )


def test_energy_to_momentum_rejects_energy_below_mass() -> None:
    with pytest.raises(
        ValueError,
        match="rest mass",
    ):
        energy_to_momentum(
            [100.0],
            Particle.MUON,
        )


def test_momentum_to_energy_rejects_negative_momentum() -> None:
    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        momentum_to_energy(
            [-1.0],
            Particle.ELECTRON,
        )


def test_particle_mass_rejects_unknown_label() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown particle labels",
    ):
        particle_mass([0, 3])


def test_kinetic_energy_momentum_round_trip() -> None:
    labels = np.array([0, 1, 2])
    expected_kinetic_energy = np.array([10.0, 100.0, 500.0])

    momentum = kinetic_energy_to_momentum(
        expected_kinetic_energy,
        labels,
    )
    actual_kinetic_energy = momentum_to_kinetic_energy(
        momentum,
        labels,
    )

    np.testing.assert_allclose(
        actual_kinetic_energy,
        expected_kinetic_energy,
    )


def test_zero_kinetic_energy_gives_zero_momentum() -> None:
    actual = kinetic_energy_to_momentum(
        [0.0, 0.0, 0.0],
        [0, 1, 2],
    )

    np.testing.assert_allclose(actual, [0.0, 0.0, 0.0])


def test_negative_kinetic_energy_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        kinetic_energy_to_momentum(
            [-1.0],
            Particle.ELECTRON,
        )
