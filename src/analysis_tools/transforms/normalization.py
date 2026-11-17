from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class MinMaxTransform:
    """

    A transform that maps values from a physical range to a normalized range.

    Parameters:
        minimum (ArrayLike): The minimum value of the physical range.
        maximum (ArrayLike): The maximum value of the physical range.
        normalized_minimum (ArrayLike): The minimum value of the normalized range.
        normalized_maximum (ArrayLike): The maximum value of the normalized range.

    Attributes:
        minimum (ArrayLike): The minimum value of the physical range.
        maximum (ArrayLike): The maximum value of the physical range.
        normalized_minimum (ArrayLike): The minimum value of the normalized range.
        normalized_maximum (ArrayLike): The maximum value of the normalized range.
    """

    minimum: ArrayLike
    maximum: ArrayLike
    normalized_minimum: ArrayLike = 0.0
    normalized_maximum: ArrayLike = 1.0

    def __post_init__(self) -> None:
        minimum = np.asarray(self.minimum, dtype=float)
        maximum = np.asarray(self.maximum, dtype=float)

        if minimum.shape != maximum.shape:
            raise ValueError("mimimum and maximum must have the same shape.")
        if np.any(maximum <= minimum):
            raise ValueError("Every maximum must be greater than minimum.")

        if self.normalized_maximum <= self.normalized_minimum:
            raise ValueError(
                "normalized_maximum must be greater than normalized_minimum."
            )

        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    def forward(self, values: ArrayLike) -> NDArray[np.float64]:
        """

        Transform physical values to the normalized range.

        Parameters
        ----------
        values: ArrayLike
            The physical values to be transformed.

        Returns
        -------
        NDArray[np.float64]
            The transformed values in the normalized range.
        """
        values = np.asarray(values, dtype=float)

        physical_range = self.maximum - self.minimum
        normalized_range = self.normalized_maximum - self.normalized_minimum

        return (
            self.normalized_minimum
            + (values - self.minimum) * normalized_range / physical_range
        )

    def inverse(self, values: ArrayLike) -> NDArray[np.float64]:
        """
        Transform normalized values to the physical range.

        Parameters
        ----------
        values: ArrayLike
            The normalized values to be transformed.

        Returns
        -------
        NDArray[np.float64]
            The transformed values in the physical range.
        """
        values = np.asarray(values, dtype=float)

        physical_range = self.maximum - self.minimum
        normalized_range = self.normalized_maximum - self.normalized_minimum

        return (
            self.minimum
            + (values - self.normalized_minimum) * physical_range / normalized_range
        )
