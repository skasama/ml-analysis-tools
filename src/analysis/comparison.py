from .data import align_by_event_id


def compare_methods(results, metric_function, *, align=True):
    """Apply one metric function with optional event-ID alignment."""
    results = tuple(results)
    if align:
        results = align_by_event_id(*results)
    return {
        result.method: metric_function(result.predictions, result.targets)
        for result in results
    }


def compare_metric(results, metric_function, *, align=True):
    """Return one scalar metric per method."""
    return compare_methods(results, metric_function, align=align)
