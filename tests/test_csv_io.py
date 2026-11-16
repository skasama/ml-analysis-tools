import numpy as np
import pytest

from analysis_tools.io.csv import load_csv, save_csv


def test_save_and_load_csv(tmp_path) -> None:
    expected = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    path = tmp_path / "results.csv"

    save_csv(
        path,
        expected,
        column_names=("bias", "resolution"),
    )
    actual = load_csv(path)

    np.testing.assert_allclose(actual["bias"], [1.0, 3.0])
    np.testing.assert_allclose(actual["resolution"], [2.0, 4.0])


def test_save_csv_without_header(tmp_path) -> None:
    expected = np.array([1.0, 2.0, 3.0])
    path = tmp_path / "values.csv"

    save_csv(path, expected)
    actual = load_csv(path, names=False)

    np.testing.assert_allclose(actual, expected)


def test_save_csv_rejects_invalid_dimensions(tmp_path) -> None:
    values = np.ones((2, 2, 2))

    with pytest.raises(ValueError, match="one- or two-dimensional"):
        save_csv(tmp_path / "values.csv", values)


def test_save_csv_rejects_wrong_column_count(tmp_path) -> None:
    values = np.ones((3, 2))

    with pytest.raises(ValueError, match="Expected 2 column names"):
        save_csv(
            tmp_path / "values.csv",
            values,
            column_names=("value",),
        )


def test_load_csv_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "missing.csv")
