import matplotlib.pyplot as plt
import numpy as np
import pytest

from analysis_tools.analysis.evaluation import (
    evaluate,
    evaluate_dependency,
)
from analysis_tools.plotting import plot_evaluation


def test_evaluate_mean() -> None:
    result = evaluate(
        [1.0, 2.0, np.nan, 3.0],
        metric=np.mean,
    )

    assert result.value == pytest.approx(2.0)


def test_evaluate_with_error() -> None:
    result = evaluate(
        [1.0, 2.0, 3.0],
        metric=np.mean,
        error=lambda values: np.std(values),
    )

    assert result.value == pytest.approx(2.0)
    assert result.error == pytest.approx(np.sqrt(2.0 / 3.0))


def test_evaluate_dependency() -> None:
    result = evaluate_dependency(
        dependency=[10.0, 20.0, 30.0, 40.0],
        values=[1.0, 2.0, 3.0, 4.0],
        edges=[0.0, 25.0, 50.0],
        metric=np.mean,
        error=lambda values: np.std(values),
    )

    np.testing.assert_allclose(result.centers, [12.5, 37.5])
    np.testing.assert_allclose(result.values, [1.5, 3.5])
    np.testing.assert_allclose(result.counts, [2, 2])
    assert result.errors is not None


def test_evaluate_dependency_counts_only_finite_metric_events() -> None:
    result = evaluate_dependency(
        dependency=[10.0, 20.0, 30.0, 40.0],
        values=[1.0, np.nan, 3.0, 4.0],
        edges=[0.0, 25.0, 50.0],
        minimum_events=2,
    )

    np.testing.assert_array_equal(result.counts, [1, 2])
    assert np.isnan(result.values[0])
    assert result.values[1] == pytest.approx(3.5)


def test_evaluate_dependency_supports_asymmetric_errors() -> None:
    result = evaluate_dependency(
        dependency=[10.0, 20.0, 30.0, 40.0],
        values=[1.0, 3.0, 2.0, 6.0],
        edges=[0.0, 25.0, 50.0],
        metric=np.mean,
        error=lambda values: (
            float(np.mean(values) - np.min(values)),
            float(np.max(values) - np.mean(values)),
        ),
    )

    assert result.errors is not None
    assert result.errors.shape == (2, 2)
    np.testing.assert_allclose(result.errors[:, 0], [1.0, 1.0])
    np.testing.assert_allclose(result.errors[:, 1], [2.0, 2.0])


def test_evaluate_dependency_accepts_numpy_asymmetric_error() -> None:
    result = evaluate_dependency(
        dependency=[10.0, 20.0],
        values=[1.0, 3.0],
        edges=[0.0, 30.0],
        error=lambda values: np.array([0.5, 1.5]),
    )

    assert result.errors is not None
    np.testing.assert_allclose(result.errors[:, 0], [0.5, 1.5])


def test_evaluate_dependency_rejects_mixed_error_shapes() -> None:
    def error(values):
        if np.mean(values) < 3.0:
            return 1.0
        return 1.0, 1.0

    with pytest.raises(ValueError, match="same shape"):
        evaluate_dependency(
            dependency=[10.0, 20.0, 30.0, 40.0],
            values=[1.0, 2.0, 4.0, 5.0],
            edges=[0.0, 25.0, 50.0],
            error=error,
        )


@pytest.mark.parametrize(
    ("values", "metric", "message"),
    [
        ([[1.0], [2.0]], np.mean, "one-dimensional"),
        ([np.nan, np.inf], np.mean, "no finite events"),
        ([1.0, 2.0], lambda values: values, "one scalar"),
    ],
)
def test_evaluate_validates_input_and_metric_output(
    values,
    metric,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate(values, metric=metric)


def test_evaluate_dependency_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same number"):
        evaluate_dependency(
            dependency=[1.0, 2.0],
            values=[1.0],
            edges=[0.0, 3.0],
        )


def test_plot_evaluation_on_existing_axes() -> None:
    result = evaluate_dependency(
        dependency=[10.0, 20.0, 30.0],
        values=[1.0, 2.0, 3.0],
        edges=[0.0, 20.0, 40.0],
    )
    figure, axis = plt.subplots()

    returned = plot_evaluation(
        result,
        ax=axis,
        label="Model",
    )

    assert returned is axis
    assert axis.figure is figure

    plt.close(figure)


def test_plot_evaluation_with_count_axes() -> None:
    result = evaluate_dependency(
        dependency=[10.0, 20.0, 30.0],
        values=[1.0, 2.0, 3.0],
        edges=[0.0, 20.0, 40.0],
    )

    metric_ax, count_ax = plot_evaluation(
        result,
        show_counts=True,
        label="Model",
    )

    assert metric_ax.figure is count_ax.figure
    assert count_ax.get_ylabel() == "Events"
    plt.close(metric_ax.figure)


def test_plot_evaluation_requires_count_axes_with_external_axes() -> None:
    result = evaluate_dependency(
        dependency=[10.0, 20.0],
        values=[1.0, 2.0],
        edges=[0.0, 30.0],
    )
    figure, axis = plt.subplots()

    with pytest.raises(ValueError, match="count_ax is required"):
        plot_evaluation(
            result,
            ax=axis,
            show_counts=True,
        )

    plt.close(figure)
