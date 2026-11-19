from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class AsymmetricError:
    """Lower and upper parameter errors from MINOS."""

    lower: float
    upper: float
    valid: bool


@dataclass(frozen=True)
class FitResult:
    """Result of a Minuit fit."""

    parameter_names: tuple[str, ...]
    values: dict[str, float]
    errors: dict[str, float]

    covariance: NDArray[np.float64] | None
    minos_errors: dict[str, AsymmetricError]

    minimum_value: float
    valid: bool
    accurate_covariance: bool
    reached_call_limit: bool
    above_maximum_edm: bool

    number_of_data_points: int
    number_of_free_parameters: int

    @property
    def degrees_of_freedom(self) -> int:
        """Return the number of degrees of freedom."""
        return self.number_of_data_points - self.number_of_free_parameters

    @property
    def reduced_minimum_value(self) -> float:
        """Return the minimum cost divided by degrees of freedom."""
        if self.degrees_of_freedom <= 0:
            return float("nan")

        return self.minimum_value / self.degrees_of_freedom

    def value(self, name: str) -> float:
        """Return one fitted parameter value."""
        return self.values[name]

    def error(self, name: str) -> float:
        """Return one symmetric parameter error."""
        return self.errors[name]
