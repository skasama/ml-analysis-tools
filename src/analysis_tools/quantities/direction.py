from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from analysis_tools.data.arrays import as_vector_array, validate_same_length
from analysis_tools.metrics.direction import (
    direction_errors,
    wrapped_angle_residuals,
)
from analysis_tools.transforms.coordinates import angles_from_direction


@dataclass(frozen=True)
class DirectionQuantities:
    """Event-level direction quantities; angles use degrees by default."""

    predictions: NDArray[np.float64]
    targets: NDArray[np.float64]
    opening_angles: NDArray[np.float64]
    prediction_angles: NDArray[np.float64]
    target_angles: NDArray[np.float64]
    theta_residuals: NDArray[np.float64]
    phi_residuals: NDArray[np.float64]

    @property
    def event_count(self) -> int:
        return len(self.targets)

    @property
    def angular_errors(self) -> NDArray[np.float64]:
        """Alias for opening angles, in degrees by default."""
        return self.opening_angles

    @property
    def prediction_theta(self) -> NDArray[np.float64]:
        return self.prediction_angles[:, 0]

    @property
    def prediction_phi(self) -> NDArray[np.float64]:
        return self.prediction_angles[:, 1]

    @property
    def target_theta(self) -> NDArray[np.float64]:
        return self.target_angles[:, 0]

    @property
    def target_phi(self) -> NDArray[np.float64]:
        return self.target_angles[:, 1]

    def select(
        self,
        selection: ArrayLike,
    ) -> "DirectionQuantities":
        selection = np.asarray(selection)
        return DirectionQuantities(
            predictions=self.predictions[selection],
            targets=self.targets[selection],
            opening_angles=self.opening_angles[selection],
            prediction_angles=self.prediction_angles[selection],
            target_angles=self.target_angles[selection],
            theta_residuals=self.theta_residuals[selection],
            phi_residuals=self.phi_residuals[selection],
        )


def calculate_direction_quantities(
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    degrees: bool = True,
) -> DirectionQuantities:
    """Calculate direction quantities.

    By default, theta, phi, their residuals, and opening angles are in degrees.
    Phi residuals use the shortest signed difference in [-180, 180] degrees.
    Pass ``degrees=False`` to return all angles and residuals in radians.
    """
    prediction_array = as_vector_array(
        predictions,
        size=3,
        name="predictions",
    ).astype(float)
    target_array = as_vector_array(
        targets,
        size=3,
        name="targets",
    ).astype(float)

    validate_same_length(
        prediction_array,
        target_array,
        names=("predictions", "targets"),
    )

    prediction_angles = angles_from_direction(prediction_array)
    target_angles = angles_from_direction(target_array)
    theta_residuals = prediction_angles[:, 0] - target_angles[:, 0]
    phi_residuals = wrapped_angle_residuals(
        prediction_angles[:, 1],
        target_angles[:, 1],
    )

    if degrees:
        prediction_angles = np.degrees(prediction_angles)
        target_angles = np.degrees(target_angles)
        theta_residuals = np.degrees(theta_residuals)
        phi_residuals = np.degrees(phi_residuals)

    return DirectionQuantities(
        predictions=prediction_array,
        targets=target_array,
        opening_angles=direction_errors(
            prediction_array,
            target_array,
            degrees=degrees,
        ),
        prediction_angles=prediction_angles,
        target_angles=target_angles,
        theta_residuals=theta_residuals,
        phi_residuals=phi_residuals,
    )
