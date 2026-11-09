from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MethodResult:
    """Predictions and targets for one method on an identified event sample."""

    method: str
    task: str
    predictions: np.ndarray
    targets: np.ndarray
    event_ids: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.method:
            raise ValueError("method must not be empty")
        if not self.task:
            raise ValueError("task must not be empty")

        predictions = np.asarray(self.predictions)
        targets = np.asarray(self.targets)
        event_ids = np.asarray(self.event_ids)
        if len(predictions) != len(targets) or len(predictions) != len(event_ids):
            raise ValueError(
                "predictions, targets, and event_ids must have the same length"
            )
        if len(np.unique(event_ids)) != len(event_ids):
            raise ValueError("event_ids must be unique")
        if not np.all(np.isfinite(predictions)) or not np.all(np.isfinite(targets)):
            raise ValueError("predictions or targets contain non-finite values")

    @property
    def size(self) -> int:
        return len(self.event_ids)


def align_results(*results: MethodResult) -> tuple[MethodResult, ...]:
    """Validate that results describe the same ordered event sample."""
    if not results:
        raise ValueError("at least one result is required")
    reference = results[0].event_ids
    for result in results[1:]:
        if not np.array_equal(reference, result.event_ids):
            raise ValueError(
                "event_ids differ between methods; align results before comparing"
            )
    return results


def align_by_event_id(*results: MethodResult) -> tuple[MethodResult, ...]:
    """Return results reordered to the common event-ID intersection."""
    if not results:
        raise ValueError("at least one result is required")
    common = set(results[0].event_ids.tolist())
    for result in results[1:]:
        common.intersection_update(result.event_ids.tolist())
    if not common:
        raise ValueError("methods have no common event IDs")

    ordered_ids = np.asarray(
        [event_id for event_id in results[0].event_ids if event_id in common],
        dtype=results[0].event_ids.dtype,
    )
    aligned = []
    for result in results:
        positions = {event_id: index for index, event_id in enumerate(result.event_ids)}
        indices = np.asarray([positions[event_id] for event_id in ordered_ids])
        aligned.append(
            MethodResult(
                method=result.method,
                task=result.task,
                predictions=result.predictions[indices],
                targets=result.targets[indices],
                event_ids=ordered_ids,
                metadata=result.metadata,
            )
        )
    return tuple(aligned)
