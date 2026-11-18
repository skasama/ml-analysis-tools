import numpy as np
from numpy.typing import ArrayLike, NDArray

from analysis_tools.physics.particles import PARTICLE_MASS_MEV, Particle


def energy_to_momentum(
    energy: ArrayLike,
    particle: int | Particle | ArrayLike,
) -> NDArray[np.float64]:
    """

    Convert total energy in MeV to momentum to MeV/c

    Parameters
    ----------
    energy : ArrayLike
        Total energy in MeV. Must be greater than or equal to the particle mass.
    particle : int | Particle | ArrayLike
        Particle label(s) as integer or Particle enum.
        Must be compatible with energy shape.

    Returns
    -------
    NDArray[np.float64]
        Momentum in MeV/c, same shape as energy and particle.
    """
    energy_array = np.asarray(energy, dtype=float)
    mass = particle_mass(particle)

    try:
        energy_array, mass = np.broadcast_arrays(
            energy_array,
            mass,
        )
    except ValueError as error:
        raise ValueError(
            "energy and particle labels are not broadcast-compatible."
        ) from error

    if not np.all(np.isfinite(energy_array)):
        raise ValueError("energy must contain only finite values.")

    if np.any(energy_array < mass):
        raise ValueError(
            "Total energy must be greater than or equal to the particle rest mass."
        )

    return np.sqrt(
        np.maximum(
            energy_array**2 - mass**2,
            0.0,
        )
    )


def momentum_to_energy(
    momentum: ArrayLike,
    particle: int | Particle | ArrayLike,
) -> NDArray[np.float64]:
    """

    Convert momentum in MeV/c to total energy in MeV.

    Parameters
    ----------
    momentum : ArrayLike
        Momentum in MeV/c. Must be non-negative.
    particle : int | Particle | ArrayLike
        Particle label(s) as integer or Particle enum.
        Must be compatible with momentum shape.

    Returns
    -------
    NDArray[np.float64]
        Total energy in MeV, same shape as momentum and particle.
    """
    momentum_array = np.asarray(momentum, dtype=float)
    mass = particle_mass(particle)

    try:
        momentum_array, mass = np.broadcast_arrays(
            momentum_array,
            mass,
        )
    except ValueError as error:
        raise ValueError(
            "Momentum and particle labels are not broadcast-compatible."
        ) from error

    if not np.all(np.isfinite(momentum_array)):
        raise ValueError("Momentum must contain only infinite values.")
    if np.any(momentum_array < 0.0):
        raise ValueError("momentum must be non-negative.")

    return np.sqrt(momentum_array**2 + mass**2)


def particle_mass(particle: int | Particle | ArrayLike) -> NDArray[np.float64]:
    """

    Return particle masses to MeV/c^2 from integer labels.

    Parameters
    ----------
    particle : int | Particle | ArrayLike
        Particle label(s) as integer or Particle enum.
        Must be compatible with the shape of the returned array.

    Returns
    -------
    NDArray[np.float64]
        Particle masses in MeV/c^2, same shape as the input particle labels.
    """
    labels = np.asarray(particle)

    integer_labels = labels.astype(np.int64)

    if not np.array_equal(labels, integer_labels):
        raise ValueError("particle labels must contain integer values.")

    masses = np.empty(
        integer_labels.shape,
        dtype=float,
    )

    known_mask = np.zeros(
        integer_labels.shape,
        dtype=bool,
    )

    for label, mass in PARTICLE_MASS_MEV.items():
        selected = integer_labels == int(label)
        masses[selected] = mass
        known_mask |= selected

    if not np.all(known_mask):
        unknown = np.unique(integer_labels[~known_mask]).tolist()

        raise ValueError(f"Unknown particle labels: {unknown}")

    return masses


def kinetic_energy_to_momentum(
    kinetic_energy: ArrayLike,
    particle: int | Particle | ArrayLike,
) -> NDArray[np.float64]:
    """

    Convert kinetic energy in MeV to momentum in MeV/c.

    Parameters
    ----------
    kinetic_energy : ArrayLike
        Kinetic energy in MeV. Must be non-negative.
    particle : int | Particle | ArrayLike
        Particle label(s) as integer or Particle enum.
        Must be compatible with kinetic_energy shape.

    Returns
    -------
    NDArray[np.float64]
        Momentum in MeV/c, same shape as kinetic_energy and particle.
    """
    kinetic_energy_array = np.asarray(
        kinetic_energy,
        dtype=float,
    )
    mass = particle_mass(particle)

    try:
        kinetic_energy_array, mass = np.broadcast_arrays(
            kinetic_energy_array,
            mass,
        )
    except ValueError as error:
        raise ValueError("kinetic_energy must contain only finite values.") from error
    if not np.all(np.isfinite(kinetic_energy_array)):
        raise ValueError("kinetic_energy must contain only finite values.")
    if np.any(kinetic_energy_array < 0.0):
        raise ValueError("kinetic_energy must be non-negative.")

    return np.sqrt(kinetic_energy_array * (kinetic_energy_array + 2.0 * mass))


def momentum_to_kinetic_energy(
    momentum: ArrayLike,
    particle: int | Particle | ArrayLike,
) -> NDArray[np.float64]:
    """

    Convert momentum in MeV/c to kinetic energy in MeV.

    Parameters
    ----------
    momentum : ArrayLike
        Momentum in MeV/c. Must be non-negative.
    particle : int | Particle | ArrayLike
        Particle label(s) as integer or Particle enum.
        Must be compatible with momentum shape.

    Returns
    -------
    NDArray[np.float64]
        Kinetic energy in MeV, same shape as momentum and particle.
    """
    total_energy = momentum_to_energy(
        momentum,
        particle,
    )
    mass = particle_mass(particle)

    return total_energy - mass
