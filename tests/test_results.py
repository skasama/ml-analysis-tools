import numpy as np
import pytest

from analysis.baselines import constant_prediction
from analysis.comparison import compare_methods
from analysis.metrics import (
    classification_metrics,
    correlation,
    regression_metrics,
    residual_profile,
    resolution68,
    vertex_metrics,
)
from analysis.results import MethodResult, align_by_event_id, align_results
from analysis.uncertainty import bootstrap_metric, profile_uncertainty


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


def test_comparison_aligns_methods_before_calculating_metrics():
    first = result("ml", [1, 2], [1.0, 2.0], [1.0, 2.0])
    second = result("baseline", [2, 1], [4.0, 3.0], [2.0, 1.0])
    metrics = compare_methods(
        (first, second),
        lambda predictions, targets: float(np.mean(np.abs(predictions - targets))),
    )
    assert metrics == {"ml": 0.0, "baseline": 2.0}


def test_comparison_can_evaluate_independent_event_sets():
    first = result("ml", [1, 2], [1.0, 2.0], [1.0, 2.0])
    second = result("baseline", [10, 20, 30], [4.0, 3.0, 5.0], [2.0, 1.0, 4.0])
    metrics = compare_methods(
        (first, second),
        regression_metrics,
        align=False,
    )
    assert metrics["ml"]["n_events"] == 2
    assert metrics["baseline"]["n_events"] == 3


def test_bootstrap_metric_returns_confidence_interval():
    result = bootstrap_metric(
        [0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0],
        lambda predictions, targets: float(np.mean(predictions - targets)),
        n_samples=20,
        seed=1,
    )
    assert result["estimate"] == pytest.approx(1.0)
    assert result["lower"] <= result["estimate"] <= result["upper"]


def test_constant_prediction_uses_target_mean():
    np.testing.assert_array_equal(constant_prediction([1.0, 3.0]), [2.0, 2.0])


def test_profile_uncertainty_returns_bias_and_resolution_intervals():
    result = profile_uncertainty(
        [0.0, 0.1, 1.0, 1.1],
        [1.0, 1.2, 2.0, 2.2],
        n_bins=2,
        n_samples=20,
        seed=1,
    )
    assert result["bias"].shape == (2, 2)
    assert result["resolution"].shape == (2, 2)
    assert np.all(result["bias"][0] <= result["bias"][1])


def test_correlation_supports_pearson_and_spearman():
    assert correlation([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert correlation([1, 3, 2], [10, 30, 20], method="spearman") == pytest.approx(1.0)


def test_vertex_metrics_include_wall_and_towall():
    predictions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    targets = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    directions = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    metrics = vertex_metrics(predictions, targets, directions=directions)

    assert {"r", "wall", "towall", "error_3d"} <= metrics.keys()
