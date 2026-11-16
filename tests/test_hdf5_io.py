import h5py
import numpy as np
import pytest

from analysis_tools.io.hdf5 import (
    list_hdf5_datasets,
    load_hdf5,
)


@pytest.fixture
def hdf5_path(tmp_path):
    path = tmp_path / "data.hy"

    with h5py.File(path, "w") as file:
        file.create_dataset("multi/labels", data=[0, 1, 2])
        file.create_dataset(
            "multi/positions",
            data=np.arange(9).reshape(3, 3),
        )
        file.create_dataset("fitqun/e_1rnll", data=[1.0, 2.0, 3.0])

    return path


def test_list_hdf5_datasets(hdf5_path) -> None:
    datasets = list_hdf5_datasets(hdf5_path)

    assert datasets == [
        "fitqun/e_1rnll",
        "multi/labels",
        "multi/positions",
    ]


def test_load_hdf5(hdf5_path) -> None:
    arrays = load_hdf5(
        hdf5_path,
        keys=["multi/labels", "multi/positions"],
    )

    np.testing.assert_array_equal(arrays["multi/labels"], [0, 1, 2])
    assert arrays["multi/positions"].shape == (3, 3)


def test_load_hdf5_rejects_missing_dataset(hdf5_path) -> None:
    with pytest.raises(KeyError, match="missing"):
        load_hdf5(hdf5_path, keys=["missing"])


def test_load_hdf5_requires_keys(hdf5_path) -> None:
    with pytest.raises(ValueError, match="At least one"):
        load_hdf5(hdf5_path, keys=[])


def test_load_hdf5_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_hdf5(tmp_path / "missing.hy", keys=["data"])
