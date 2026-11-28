from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from analysis_tools.data.arrays import validate_same_length
from analysis_tools.metrics.energy import (
    energy_residuals,
    fractional_energy_residuals,
)


@dataclass(frozen=True)
class EnergyQuantities:
    """Event-level quantities for energy or momentum regression."""

    predictions: NDArray[np.float64]
    targets: NDArray[np.float64]
    residuals: NDArray[np.float64]
    relative_residuals: NDArray[np.float64]

    @property
    def event_count(self) -> int:
        return len(self.targets)

    def select(
        self,
        selection: ArrayLike,
    ) -> "EnergyQuantities":
        return calculate_energy_quantities(
            predictions=self.predictions[selection],
            targets=self.targets[selection],
        )


def calculate_energy_quantities(
    predictions: ArrayLike,
    targets: ArrayLike,
) -> EnergyQuantities:
    """Calculate event-level energy or momentum quantities."""
    prediction_array = np.asarray(predictions, dtype=float)
    target_array = np.asarray(targets, dtype=float)

    if prediction_array.ndim == 2 and prediction_array.shape[1:] == (1,):
        prediction_array = prediction_array[:, 0]
    if target_array.ndim == 2 and target_array.shape[1:] == (1,):
        target_array = target_array[:, 0]

    if prediction_array.ndim != 1 or target_array.ndim != 1:
        raise ValueError(
            "predictions and targets must be one-dimensional or have singleton columns."
        )

    validate_same_length(
        prediction_array,
        target_array,
        names=("predictions", "targets"),
    )

    return EnergyQuantities(
        predictions=prediction_array,
        targets=target_array,
        residuals=energy_residuals(prediction_array, target_array),
        relative_residuals=fractional_energy_residuals(
            prediction_array,
            target_array,
        ),
    )
