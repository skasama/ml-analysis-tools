from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from analysis_tools.data import (
    as_integer_array,
    as_scalar_array,
    as_vector_array,
    validate_same_length,
)
from analysis_tools.io.hdf5 import load_hdf5


def fitqun_electron_muon_score(
    electron_nll: ArrayLike,
    muon_nll: ArrayLike,
    electron_momentum: ArrayLike,
    *,
    momentum_weight: float = 0.2,
) -> NDArray[np.float64]:
    """Calculate the fiTQun electron-versus-muon score."""
    electron = as_scalar_array(
        electron_nll,
        name="electron_nll",
    ).astype(float)
    muon = as_scalar_array(
        muon_nll,
        name="muon_nll",
    ).astype(float)
    momentum = as_scalar_array(
        electron_momentum,
        name="electron_momentum",
    ).astype(float)

    validate_same_length(
        electron,
        muon,
        momentum,
        names=("electron_nll", "muon_nll", "electron_momentum"),
    )

    return muon - electron - momentum_weight * momentum


def fitqun_electron_muon_predictions(
    electron_nll: ArrayLike,
    muon_nll: ArrayLike,
    electron_momentum: ArrayLike,
    *,
    momentum_weight: float = 0.2,
    threshold: float = 0.0,
    electron_label: int = 1,
    muon_label: int = 0,
) -> NDArray[np.int64]:
    """Calculate fiTQun electron-versus-muon predicted labels."""
    score = fitqun_electron_muon_score(
        electron_nll,
        muon_nll,
        electron_momentum,
        momentum_weight=momentum_weight,
    )

    return np.where(
        score > threshold,
        electron_label,
        muon_label,
    ).astype(np.int64)


@dataclass(frozen=True)
class FitQunKeys:
    """HDF5 dataset keys for fiTQun and truth information."""

    labels: str = "multi/labels"
    truth_positions: str = "multi/positions"
    truth_directions: str = "multi/directions"
    truth_energies: str = "multi/energies"

    electron_nll: str = "fitqun/e_1rnll"
    muon_nll: str = "fitqun/mu_1rnll"
    electron_momentum: str = "fitqun/e_1rmom"

    reconstructed_positions: str | None = None
    reconstructed_directions: str | None = None
    reconstructed_momenta: str | None = None


@dataclass(frozen=True)
class FitQunData:
    """Loaded fiTQun reconstruction and corresponding truth data."""

    labels: NDArray[np.int64]

    truth_positions: NDArray[np.float64]
    truth_directions: NDArray[np.float64]
    truth_energies: NDArray[np.float64]

    electron_nll: NDArray[np.float64]
    muon_nll: NDArray[np.float64]
    electron_momentum: NDArray[np.float64]

    reconstructed_positions: NDArray[np.float64] | None = None
    reconstructed_directions: NDArray[np.float64] | None = None
    reconstructed_momenta: NDArray[np.float64] | None = None

    @property
    def event_count(self) -> int:
        return len(self.labels)

    @property
    def electron_muon_score(self) -> NDArray[np.float64]:
        """Return the standard electron-versus-muon fiTQun score."""
        return fitqun_electron_muon_score(
            self.electron_nll,
            self.muon_nll,
            self.electron_momentum,
        )

    def electron_muon_predictions(
        self,
        *,
        electron_label: int = 1,
        muon_label: int = 0,
        threshold: float = 0.0,
    ) -> NDArray[np.int64]:
        """Classify events using the electron-muon fiTQun score."""
        return fitqun_electron_muon_predictions(
            self.electron_nll,
            self.muon_nll,
            self.electron_momentum,
            threshold=threshold,
            electron_label=electron_label,
            muon_label=muon_label,
        )


def load_fitqun(
    path: str | Path,
    *,
    keys: FitQunKeys | None = None,
) -> FitQunData:
    """Load fiTQun reconstruction and truth arrays from HDF5."""
    keys = keys or FitQunKeys()

    required_keys = [
        keys.labels,
        keys.truth_positions,
        keys.truth_directions,
        keys.truth_energies,
        keys.electron_nll,
        keys.muon_nll,
        keys.electron_momentum,
    ]

    optional_keys = [
        key
        for key in (
            keys.reconstructed_positions,
            keys.reconstructed_directions,
            keys.reconstructed_momenta,
        )
        if key is not None
    ]

    arrays = load_hdf5(
        path,
        keys=required_keys + optional_keys,
    )

    labels = as_integer_array(
        arrays[keys.labels],
        name=keys.labels,
    )
    truth_positions = as_vector_array(
        arrays[keys.truth_positions],
        size=3,
        name=keys.truth_positions,
    ).astype(float)
    truth_directions = as_vector_array(
        arrays[keys.truth_directions],
        size=3,
        name=keys.truth_directions,
    ).astype(float)
    truth_energies = as_scalar_array(
        arrays[keys.truth_energies],
        name=keys.truth_energies,
    ).astype(float)

    electron_nll = as_scalar_array(
        arrays[keys.electron_nll],
        name=keys.electron_nll,
    ).astype(float)
    muon_nll = as_scalar_array(
        arrays[keys.muon_nll],
        name=keys.muon_nll,
    ).astype(float)
    electron_momentum = as_scalar_array(
        arrays[keys.electron_momentum],
        name=keys.electron_momentum,
    ).astype(float)

    optional = _load_optional_arrays(arrays, keys)

    validate_same_length(
        labels,
        truth_positions,
        truth_directions,
        truth_energies,
        electron_nll,
        muon_nll,
        electron_momentum,
        *(array for array in optional.values() if array is not None),
    )

    return FitQunData(
        labels=labels,
        truth_positions=truth_positions,
        truth_directions=truth_directions,
        truth_energies=truth_energies,
        electron_nll=electron_nll,
        muon_nll=muon_nll,
        electron_momentum=electron_momentum,
        reconstructed_positions=optional["positions"],
        reconstructed_directions=optional["directions"],
        reconstructed_momenta=optional["momenta"],
    )


def _load_optional_arrays(
    arrays: dict[str, NDArray],
    keys: FitQunKeys,
) -> dict[str, NDArray[np.float64] | None]:
    positions = None
    directions = None
    momenta = None

    if keys.reconstructed_positions is not None:
        positions = as_vector_array(
            arrays[keys.reconstructed_positions],
            size=3,
            name=keys.reconstructed_positions,
        ).astype(float)

    if keys.reconstructed_directions is not None:
        directions = as_vector_array(
            arrays[keys.reconstructed_directions],
            size=3,
            name=keys.reconstructed_directions,
        ).astype(float)

    if keys.reconstructed_momenta is not None:
        momenta = as_scalar_array(
            arrays[keys.reconstructed_momenta],
            name=keys.reconstructed_momenta,
        ).astype(float)

    return {
        "positions": positions,
        "directions": directions,
        "momenta": momenta,
    }
