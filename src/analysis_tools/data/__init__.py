from analysis_tools.data.alignment import (
    find_duplicate_keys,
    match_event_indices,
    validate_aligned,
    validate_matched_samples,
    validate_unique_keys,
)
from analysis_tools.data.arrays import (
    as_integer_array,
    as_scalar_array,
    as_vector_array,
    validate_same_length,
)
from analysis_tools.data.binning import (
    BinnedSelection,
    make_binned_selection,
)
from analysis_tools.data.hits import (
    hit_counts_from_indices,
)
from analysis_tools.data.models import (
    EventIndex,
    IndependentComparisonSample,
    MatchedComparisonSample,
)
from analysis_tools.data.selection import (
    CutFlowEntry,
    EventSelection,
    apply_mask,
    apply_mask_to_mapping,
    combine_masks,
    range_mask,
)

__all__ = [
    "EventIndex",
    "IndependentComparisonSample",
    "MatchedComparisonSample",
    "find_duplicate_keys",
    "match_event_indices",
    "validate_aligned",
    "validate_matched_samples",
    "validate_unique_keys",
    "apply_mask",
    "apply_mask_to_mapping",
    "combine_masks",
    "range_mask",
    "as_scalar_array",
    "as_vector_array",
    "validate_same_length",
    "BinnedSelection",
    "make_binned_selection",
    "as_integer_array",
    "hit_counts_from_indices",
    "CutFlowEntry",
    "EventSelection",
]
