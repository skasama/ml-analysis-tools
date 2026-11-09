import numpy as np
import pytest

from analysis.metrics import (
    classification_metrics,
    regression_metrics,
    residual_profile,
    resolution68,
)
from analysis.results import MethodResult, align_by_event_id, align_results


def result(method, event_ids, predictions, targets):
    return MethodResult(
        method=method,
        task="regression",
        predictions=np.asarray(predictions),
        targets=np.asarray(targets),
        event_ids=np.asarray(event_ids),
    )


def test_method_result_rejects_duplicate_event_ids():
    with pytest.raises(ValueError, match="unique"):
        result("ml", [1, 1], [0.0, 1.0], [0.0, 1.0])


def test_align_by_event_id_reorders_common_events():
    first, second = align_by_event_id(
        result("ml", [10, 20], [1.0, 2.0], [1.0, 2.0]),
        result("baseline", [20, 10], [4.0, 3.0], [2.0, 1.0]),
    )
    np.testing.assert_array_equal(first.event_ids, [10, 20])
    np.testing.assert_array_equal(second.event_ids, [10, 20])
    np.testing.assert_array_equal(second.predictions, [3.0, 4.0])


def test_align_results_rejects_different_order():
    with pytest.raises(ValueError, match="differ"):
        align_results(
            result("ml", [1, 2], [0.0, 1.0], [0.0, 1.0]),
            result("baseline", [2, 1], [0.0, 1.0], [0.0, 1.0]),
        )


def test_regression_metrics_include_common_quantiles():
    metrics = regression_metrics([0.0, 2.0, 4.0], [0.0, 1.0, 2.0])
    assert metrics["n_events"] == 3
    assert metrics["bias"] == pytest.approx(1.0)
    assert metrics["mae"] == pytest.approx(1.0)


def test_classification_metrics_accepts_one_hot_targets():
    metrics = classification_metrics(
        [[3.0, 0.0], [0.0, 3.0]],
        [[1.0, 0.0], [1.0, 0.0]],
    )
    assert metrics["accuracy"] == pytest.approx(0.5)


def test_resolution68_ignores_non_finite_values():
    assert resolution68([1.0, -2.0, np.nan]) == pytest.approx(1.68)


def test_residual_profile_returns_binned_statistics():
    centres, biases, resolutions, counts = residual_profile(
        [0.0, 0.5, 1.0, 1.5],
        [1.0, 1.0, 3.0, 3.0],
        n_bins=2,
    )
    np.testing.assert_allclose(centres, [0.375, 1.125])
    np.testing.assert_allclose(biases, [1.0, 3.0])
    np.testing.assert_allclose(resolutions, [1.0, 3.0])
    np.testing.assert_array_equal(counts, [2, 2])
