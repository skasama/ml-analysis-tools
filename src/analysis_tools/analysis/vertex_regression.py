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
from analysis_tools.quantities.vertex import (
    VertexQuantities,
    calculate_vertex_quantities,
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

    def to_quantities(self) -> "VertexQuantities":
        """Convert this compatibility result to event-level vertex quantities."""
        return calculate_vertex_quantities(
            predictions=self.predictions,
            targets=self.targets,
        )


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
