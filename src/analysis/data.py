"""Common data objects used by ML and non-ML analysis methods."""

from .results import MethodResult, align_by_event_id, align_results

__all__ = ["MethodResult", "align_by_event_id", "align_results"]
