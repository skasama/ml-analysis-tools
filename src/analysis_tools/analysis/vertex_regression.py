from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from analysis_tools.data.arrays import (
    as_vector_array,
    validate_same_length,
)
from analysis_tools.metrics.vertex import (
    vertex_errors,
    vertex_residuals,
)
from analysis_tools.transforms.geometry import (
    compute_wall,
    cylindrical_radius,
)


@dataclass(frozen=True)
class VertexRegressionResult:
    predictions: NDArray[np.float64]
    targets: NDArray[np.float64]
    residuals: NDArray[np.float64]
    errors: NDArray[np.float64]

    predicted_radius: NDArray[np.float64]
    target_radius: NDArray[np.float64]
    radius_residuals: NDArray[np.float64]

    predicted_wall: NDArray[np.float64]
    target_wall: NDArray[np.float64]
    wall_residuals: NDArray[np.float64]

    valid_mask: NDArray[np.bool_]
    removed_event_count: int

    @property
    def x_residuals(self) -> NDArray[np.float64]:
        return self.residuals[:, 0]

    @property
    def y_residuals(self) -> NDArray[np.float64]:
        return self.residuals[:, 1]

    @property
    def z_residuals(self) -> NDArray[np.float64]:
        return self.residuals[:, 2]

    @property
    def event_count(self) -> int:
        return len(self.errors)


def analyze_vertex_regression(
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    detector_radius: float,
    detector_half_height: float,
    remove_nonfinite: bool = True,
) -> VertexRegressionResult:
    """

    Calculate quantities used in vertex-regression analyses.

    Parameters
    ----------
    predictions : ArrayLike
        Predicted vertex positions, shape (N, 3).
    targets : ArrayLike
        True vertex positions, shape (N, 3).
    detector_radius : float
        The radius of the cylindrical detector.
    detector_half_height : float
        Half the height of the cylindrical detector.
    remove_nonfinite : bool, optional
        If True, remove events with non-finite predictions or targets.
        Default is True.

    Returns
    -------
    VertexRegressionResult
        A dataclass containing the results of the vertex regression analysis.
    """
    predictions = as_vector_array(
        predictions,
        size=3,
        name="predictions",
    ).astype(float)

    targets = as_vector_array(
        targets,
        size=3,
        name="targets",
    ).astype(float)

    original_count = validate_same_length(
        predictions,
        targets,
        names=("predictions", "targets"),
    )

    valid_mask = np.all(np.isfinite(predictions), axis=1)
    valid_mask &= np.all(np.isfinite(targets), axis=1)

    if remove_nonfinite:
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]
    elif not np.all(valid_mask):
        raise ValueError("Predictions and targets contain non-finite values.")

    if len(predictions) == 0:
        raise ValueError("No valid vertex events remain.")

    residuals = vertex_residuals(predictions, targets)
    errors = vertex_errors(predictions, targets)

    predicted_radius = cylindrical_radius(predictions)
    target_radius = cylindrical_radius(targets)

    predicted_wall = compute_wall(
        predictions,
        radius=detector_radius,
        half_height=detector_half_height,
    )
    target_wall = compute_wall(
        targets,
        radius=detector_radius,
        half_height=detector_half_height,
    )

    return VertexRegressionResult(
        predictions=predictions,
        targets=targets,
        residuals=residuals,
        errors=errors,
        predicted_radius=predicted_radius,
        target_radius=target_radius,
        radius_residuals=predicted_radius - target_radius,
        predicted_wall=predicted_wall,
        target_wall=target_wall,
        wall_residuals=predicted_wall - target_wall,
        valid_mask=valid_mask,
        removed_event_count=original_count - len(predictions),
    )


@dataclass(frozen=True)
class VertexQuantities:
    """

    Event-level quantities for vertex analysis.

    Attributes
    ----------
    predictions : np.ndarray
        The predicted vertex positions, shape (N, 3).
    targets : np.ndarray
        The true vertex positions, shape (N, 3).
    residuals : np.ndarray
        The residuals between predictions and targets, shape (N, 3).
    residual_r : np.ndarray
        The radial residuals between predictions and targets, shape (N,).
    errors_3d : np.ndarray
        The 3D errors between predictions and targets, shape (N,).
    prediction_r : np.ndarray
        The predicted radial positions, shape (N,).
    target_r : np.ndarray
        The true radial positions, shape (N,).
    """

    predictions: np.ndarray
    targets: np.ndarray

    residuals: np.ndarray
    residual_r: np.ndarray
    errors_3d: np.ndarray

    prediction_r: np.ndarray
    target_r: np.ndarray

    @property
    def event_count(self) -> int:
        """Return the number of events."""
        return len(self.targets)

    @property
    def residual_x(self) -> np.ndarray:
        """Return x residuals."""
        return self.residuals[:, 0]

    @property
    def residual_y(self) -> np.ndarray:
        """Return y residuals."""
        return self.residuals[:, 1]

    @property
    def residual_z(self) -> np.ndarray:
        """Return z residuals."""
        return self.residuals[:, 2]

    def select(self, selection: np.ndarray) -> "VertexQuantities":
        """Return quantities selected by a Boolean or index array."""
        return VertexQuantities(
            predictions=self.predictions[selection],
            targets=self.targets[selection],
            residuals=self.residuals[selection],
            residual_r=self.residual_r[selection],
            errors_3d=self.errors_3d[selection],
            prediction_r=self.prediction_r[selection],
            target_r=self.target_r[selection],
        )


def calculate_vertex_quantities(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> VertexQuantities:
    """

    Calculate event-level vertex quantities.

    Parameters
    ----------
    predictions : np.ndarray
        The predicted vertex positions, shape (N, 3).
    targets : np.ndarray
        The true vertex positions, shape (N, 3).

    Returns
    -------
    VertexQuantities
        The calculated vertex quantities.
    """

    predictions = as_vector_array(
        predictions,
        size=3,
        name="predictions",
    ).astype(float)

    targets = as_vector_array(targets, size=3, name="targets").astype(float)

    validate_same_length(predictions, targets, names=("predictions", "targets"))

    residuals = predictions - targets

    prediction_r = np.sqrt(predictions[:, 0] ** 2 + predictions[:, 1] ** 2)
    target_r = np.sqrt(targets[:, 0] ** 2 + targets[:, 1] ** 2)
    residual_r = prediction_r - target_r

    errors_3d = np.linalg.norm(residuals, axis=1)

    return VertexQuantities(
        predictions=predictions,
        targets=targets,
        residuals=residuals,
        residual_r=residual_r,
        errors_3d=errors_3d,
        prediction_r=prediction_r,
        target_r=target_r,
    )
