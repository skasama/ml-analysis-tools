from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from analysis_tools.data.arrays import (
    as_scalar_array,
    validate_same_length,
)


@dataclass(frozen=True)
class EventIndex:
    """A class to represent the index of events in a dataset."""

    file_ids: NDArray
    event_ids: NDArray

    @classmethod
    def from_arrays(
        cls,
        file_ids: ArrayLike,
        event_ids: ArrayLike,
    ) -> "EventIndex":
        """Create an EventIndex from arrays of file IDs and event IDs."""
        file_ids_array = as_scalar_array(file_ids, name="file_ids")
        event_ids_array = as_scalar_array(event_ids, name="event_ids")

        validate_same_length(
            file_ids_array,
            event_ids_array,
            names=("file_ids", "event_ids"),
        )

        return cls(
            file_ids=file_ids_array,
            event_ids=event_ids_array,
        )

    def __len__(self) -> int:
        return len(self.file_ids)

    def keys(self) -> NDArray:
        keys = np.empty(
            len(self),
            dtype=[
                ("file_id", self.file_ids.dtype),
                ("event_id", self.event_ids.dtype),
            ],
        )
        keys["file_id"] = self.file_ids
        keys["event_id"] = self.event_ids

        return keys
