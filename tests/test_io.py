import numpy as np
import pytest

from analysis.io import RunOutputs, assert_aligned, load_outputs


def write_outputs(run_dir, predictions, targets, indices=None):
    output_dir = run_dir / "outputs"
    output_dir.mkdir()
    np.save(output_dir / "preds.npy", predictions)
    np.save(output_dir / "targets.npy", targets)
    if indices is not None:
        np.save(output_dir / "indices.npy", indices)


def test_load_outputs_reads_matching_arrays(tmp_path):
    predictions = np.array([[0.2], [0.8]])
    targets = np.array([[0.0], [1.0]])
    indices = np.array([4, 8])
    write_outputs(tmp_path, predictions, targets, indices)

    result = load_outputs(tmp_path)

    assert isinstance(result, RunOutputs)
    np.testing.assert_array_equal(result.predictions, predictions)
    np.testing.assert_array_equal(result.targets, targets)
    np.testing.assert_array_equal(result.indices, indices)


def test_load_outputs_allows_optional_missing_indices(tmp_path):
    write_outputs(tmp_path, [[0.2]], [[0.0]])

    result = load_outputs(tmp_path)

    assert result.indices is None


@pytest.mark.parametrize(
    ("predictions", "targets", "message"),
    [
        ([[0.2], [0.8]], [[0.0]], "Event count mismatch"),
        ([[np.nan]], [[0.0]], "non-finite"),
    ],
)
def test_load_outputs_rejects_invalid_arrays(tmp_path, predictions, targets, message):
    write_outputs(tmp_path, predictions, targets)

    with pytest.raises(ValueError, match=message):
        load_outputs(tmp_path)


def test_load_outputs_requires_indices_when_requested(tmp_path):
    write_outputs(tmp_path, [[0.2]], [[0.0]])

    with pytest.raises(FileNotFoundError, match="Indices file"):
        load_outputs(tmp_path, require_indices=True)


def test_assert_aligned_checks_event_counts_and_indices():
    first = RunOutputs(np.zeros((2, 1)), np.zeros((2, 1)), np.array([1, 2]))
    second = RunOutputs(np.zeros((2, 1)), np.zeros((2, 1)), np.array([1, 2]))
    different = RunOutputs(np.zeros((2, 1)), np.zeros((2, 1)), np.array([1, 3]))

    assert_aligned(first, second, require_indices=True)
    with pytest.raises(ValueError, match="indices.npy differs"):
        assert_aligned(first, different)

    mismatched = RunOutputs(np.zeros((1, 1)), np.zeros((1, 1)), None)
    with pytest.raises(ValueError, match="different event counts"):
        assert_aligned(first, mismatched)
