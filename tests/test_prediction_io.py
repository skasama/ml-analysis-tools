import numpy as np
import pytest

from analysis_tools.io import (
    PredictionData,
    load_prediction_data,
)


def test_load_prediction_data(
    tmp_path,
) -> None:
    predictions_path = tmp_path / "preds.npy"
    targets_path = tmp_path / "targets.npy"

    predictions = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
    )
    targets = np.array(
        [
            [0.0, 0.1, 0.2],
            [0.3, 0.4, 0.5],
        ]
    )

    np.save(
        predictions_path,
        predictions,
    )
    np.save(
        targets_path,
        targets,
    )

    data = load_prediction_data(
        predictions_path,
        targets_path,
    )

    assert isinstance(
        data,
        PredictionData,
    )
    assert data.event_count == 2
    assert data.indices is None

    np.testing.assert_array_equal(
        data.predictions,
        predictions,
    )
    np.testing.assert_array_equal(
        data.targets,
        targets,
    )


def test_load_prediction_data_with_indices(
    tmp_path,
) -> None:
    predictions_path = tmp_path / "preds.npy"
    targets_path = tmp_path / "targets.npy"
    indices_path = tmp_path / "indices.npy"

    np.save(
        predictions_path,
        [[0.1], [0.2], [0.3]],
    )
    np.save(
        targets_path,
        [[0.0], [0.1], [0.2]],
    )
    np.save(
        indices_path,
        [10, 4, 25],
    )

    data = load_prediction_data(
        predictions_path,
        targets_path,
        indices_path,
    )

    np.testing.assert_array_equal(
        data.indices,
        [10, 4, 25],
    )


def test_prediction_data_select() -> None:
    data = PredictionData(
        predictions=np.array([[1.0], [2.0], [3.0]]),
        targets=np.array([[0.0], [1.0], [2.0]]),
        indices=np.array([10, 20, 30]),
    )

    selected = data.select(np.array([True, False, True]))

    np.testing.assert_array_equal(
        selected.predictions,
        [[1.0], [3.0]],
    )
    np.testing.assert_array_equal(
        selected.targets,
        [[0.0], [2.0]],
    )
    np.testing.assert_array_equal(
        selected.indices,
        [10, 30],
    )


def test_load_prediction_data_rejects_length_mismatch(
    tmp_path,
) -> None:
    predictions_path = tmp_path / "preds.npy"
    targets_path = tmp_path / "targets.npy"

    np.save(
        predictions_path,
        [[1.0], [2.0]],
    )
    np.save(
        targets_path,
        [[1.0]],
    )

    with pytest.raises(
        ValueError,
        match="same number of events",
    ):
        load_prediction_data(
            predictions_path,
            targets_path,
        )


def test_load_prediction_data_rejects_index_length_mismatch(
    tmp_path,
) -> None:
    predictions_path = tmp_path / "preds.npy"
    targets_path = tmp_path / "targets.npy"
    indices_path = tmp_path / "indices.npy"

    np.save(
        predictions_path,
        [[1.0], [2.0]],
    )
    np.save(
        targets_path,
        [[1.0], [2.0]],
    )
    np.save(
        indices_path,
        [10],
    )

    with pytest.raises(
        ValueError,
        match="same number of events",
    ):
        load_prediction_data(
            predictions_path,
            targets_path,
            indices_path,
        )


def test_load_prediction_data_rejects_duplicate_indices(
    tmp_path,
) -> None:
    predictions_path = tmp_path / "preds.npy"
    targets_path = tmp_path / "targets.npy"
    indices_path = tmp_path / "indices.npy"

    np.save(
        predictions_path,
        [[1.0], [2.0]],
    )
    np.save(
        targets_path,
        [[1.0], [2.0]],
    )
    np.save(
        indices_path,
        [10, 10],
    )

    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        load_prediction_data(
            predictions_path,
            targets_path,
            indices_path,
        )
