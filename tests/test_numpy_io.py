import numpy as np
import pytest

from analysis_tools.io.numpy import load_npy, load_npz, save_npy, save_npz


def test_save_and_load_npy(tmp_path) -> None:
    expected = np.arange(6).reshape(2, 3)
    path = tmp_path / "array.npy"

    returned_path = save_npy(path, expected)
    actual = load_npy(path)

    assert returned_path == path
    np.testing.assert_array_equal(actual, expected)


def test_save_and_load_npz(tmp_path) -> None:
    expected = {
        "predictions": np.array([1.0, 2.0]),
        "targets": np.array([1.5, 2.5]),
    }
    path = tmp_path / "arrays.npz"

    returned_path = save_npz(path, expected)
    actual = load_npz(path)

    assert returned_path == path
    assert actual.keys() == expected.keys()

    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_load_selected_npz_keys(tmp_path) -> None:
    path = tmp_path / "results.npz"

    save_npz(
        path,
        {
            "predictions": np.array([1.0]),
            "targets": np.array([2.0]),
        },
    )

    actual = load_npz(path, keys=["predictions"])

    assert actual.keys() == {"predictions"}


def test_load_npz_rejects_missing_key(tmp_path) -> None:
    path = tmp_path / "results.npz"
    save_npz(path, {"predictions": np.array([1.0])})

    with pytest.raises(KeyError, match="targets"):
        load_npz(path, keys=["targets"])


def test_load_npy_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_npy(tmp_path / "missing.npy")


def test_save_npz_rejects_empty_mapping(tmp_path) -> None:
    with pytest.raises(ValueError, match="At least one array"):
        save_npz(tmp_path / "empty.npz", {})
