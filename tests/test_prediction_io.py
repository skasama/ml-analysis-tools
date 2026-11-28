import h5py
import numpy as np
import pytest

from analysis_tools.data import EventSelection
from analysis_tools.io import (
    MLRunConfig,
    PredictionData,
    load_ml_run,
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


def test_load_ml_run(tmp_path) -> None:
    source_path = tmp_path / "source.hy"
    predictions_path = tmp_path / "predictions.npy"
    targets_path = tmp_path / "targets.npy"
    indices_path = tmp_path / "indices.npy"

    with h5py.File(source_path, "w") as file:
        file.create_dataset(
            "multi/labels",
            data=[0, 1, 2],
        )
        file.create_dataset(
            "multi/positions",
            data=np.arange(9).reshape(3, 3),
        )

    np.save(predictions_path, [[1.0], [2.0]])
    np.save(targets_path, [[0.0], [1.0]])
    np.save(indices_path, [2, 0])

    run = load_ml_run(
        MLRunConfig(
            name="ML",
            source_hdf5=source_path,
            predictions_path=predictions_path,
            targets_path=targets_path,
            indices_path=indices_path,
            color="tab:blue",
        ),
        hdf5_keys=["multi/labels", "multi/positions"],
    )

    assert run.event_count == 2
    assert run.name == "ML"
    assert run.color == "tab:blue"
    np.testing.assert_array_equal(run.predictions.indices, [2, 0])
    np.testing.assert_array_equal(run.source["multi/labels"], [2, 0])
    np.testing.assert_array_equal(
        run.source["multi/positions"],
        np.array([[6, 7, 8], [0, 1, 2]]),
    )

    selection = EventSelection(run.event_count)
    selection.add("Selected events", np.array([False, True]))
    selected = run.select(selection.mask)

    assert selected.event_count == 1
    np.testing.assert_array_equal(selected.predictions.indices, [0])
    np.testing.assert_array_equal(selected.source["multi/labels"], [0])


def test_load_ml_run_requires_indices(tmp_path) -> None:
    config = MLRunConfig(
        name="ML",
        source_hdf5=tmp_path / "source.hy",
        predictions_path=tmp_path / "predictions.npy",
        targets_path=tmp_path / "targets.npy",
    )

    with pytest.raises(ValueError, match="indices_path is required"):
        load_ml_run(config, hdf5_keys=["multi/labels"])
