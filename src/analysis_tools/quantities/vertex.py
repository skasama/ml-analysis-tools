from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from analysis_tools.data.arrays import as_vector_array, validate_same_length


@dataclass(frozen=True)
class VertexQuantities:
    """Event-level quantities for vertex analysis."""

    predictions: NDArray[np.float64]
    targets: NDArray[np.float64]
    residuals: NDArray[np.float64]
    residual_r: NDArray[np.float64]
    errors_3d: NDArray[np.float64]
    prediction_r: NDArray[np.float64]
    target_r: NDArray[np.float64]

    @property
    def event_count(self) -> int:
        return len(self.targets)

    @property
    def residual_x(self) -> NDArray[np.float64]:
        return self.residuals[:, 0]

    @property
    def residual_y(self) -> NDArray[np.float64]:
        return self.residuals[:, 1]

    @property
    def residual_z(self) -> NDArray[np.float64]:
        return self.residuals[:, 2]

    def select(self, selection: ArrayLike) -> "VertexQuantities":
        selection = np.asarray(selection)
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
    predictions: ArrayLike,
    targets: ArrayLike,
) -> VertexQuantities:
    """Calculate event-level vertex quantities."""
    predictions_array = as_vector_array(
        predictions,
        size=3,
        name="predictions",
    ).astype(float)
    targets_array = as_vector_array(
        targets,
        size=3,
        name="targets",
    ).astype(float)

    validate_same_length(
        predictions_array,
        targets_array,
        names=("predictions", "targets"),
    )

    residuals = predictions_array - targets_array
    prediction_r = np.sqrt(predictions_array[:, 0] ** 2 + predictions_array[:, 1] ** 2)
    target_r = np.sqrt(targets_array[:, 0] ** 2 + targets_array[:, 1] ** 2)
    residual_r = prediction_r - target_r
    errors_3d = np.linalg.norm(residuals, axis=1)

    return VertexQuantities(
        predictions=predictions_array,
        targets=targets_array,
        residuals=residuals,
        residual_r=residual_r,
        errors_3d=errors_3d,
        prediction_r=prediction_r,
        target_r=target_r,
    )
