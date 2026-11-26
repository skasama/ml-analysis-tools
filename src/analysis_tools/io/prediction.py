from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from analysis_tools.data import as_integer_array
from analysis_tools.io.numpy import load_npy


@dataclass(frozen=True)
class PredictionData:
    """
    Data class for storing prediction, target, and optional source-index arrays.

    Attributes
    ----------
    predictions : np.ndarray
        Prediction array with shape (n_events, ...).
    targets : np.ndarray
        Target array with shape (n_events, ...).
    indices : NDArray[np.int64] | None
        Optional source-event indices with shape (n_events,).
    """

    predictions: np.ndarray
    targets: np.ndarray
    indices: NDArray[np.int64] | None = None

    @property
    def event_count(self) -> int:
        """Return the number of events."""
        return len(self.predictions)

    def select(
        self,
        selection: np.ndarray,
    ) -> "PredictionData":
        """

        Return data selected by an index or Boolean array.

        Parameters
        ----------
        selection : np.ndarray
            Index or Boolean array for selecting events.

        Returns
        -------
        PredictionData
            Selected prediction data.
        """
        selection = np.asarray(selection)

        selected_indices = None

        if self.indices is not None:
            selected_indices = self.indices[selection]

        return PredictionData(
            predictions=self.predictions[selection],
            targets=self.targets[selection],
            indices=selected_indices,
        )


def load_prediction_data(
    predictions_path: str | Path,
    targets_path: str | Path,
    indices_path: str | Path | None = None,
) -> PredictionData:
    """

    Load predictions, targets, and optional source indices.

    Parameters
    ----------
    predictions_path
        Path to the NumPy prediction array.
    targets_path
        Path to the NumPy target array.
    indices_path
        Optional path to source-event indices.

    Returns
    -------
    PredictionData
        Loaded event-aligned arrays.
    """
    predictions = load_npy(predictions_path)

    targets = load_npy(targets_path)

    if predictions.ndim == 0:
        raise ValueError("predictions must contain an event dimension.")

    if targets.ndim == 0:
        raise ValueError("targets must contain an event dimension.")

    if len(predictions) != len(targets):
        raise ValueError(
            "predictions and targets must have the same "
            "number of events: "
            f"{len(predictions)} != {len(targets)}."
        )

    indices = None

    if indices_path is not None:
        indices = as_integer_array(
            load_npy(indices_path),
            name="indices",
        ).astype(
            np.int64,
            copy=False,
        )

        if len(indices) != len(predictions):
            raise ValueError(
                "indices and predictions must have the same "
                "number of events: "
                f"{len(indices)} != {len(predictions)}."
            )

        if np.any(indices < 0):
            raise ValueError("indices must be non-negative.")

        if len(np.unique(indices)) != len(indices):
            raise ValueError("indices must not contain duplicates.")

    return PredictionData(
        predictions=predictions,
        targets=targets,
        indices=indices,
    )
