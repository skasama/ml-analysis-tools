import h5py
import numpy as np
import pytest

from analysis_tools.io.fitqun import (
    FitQunKeys,
    load_fitqun,
)


@pytest.fixture
def fitqun_path(tmp_path):
    path = tmp_path / "fitqun.hy"

    with h5py.File(path, "w") as file:
        file.create_dataset(
            "multi/labels",
            data=[0, 1, 1],
        )
        file.create_dataset(
            "multi/positions",
            data=np.arange(9).reshape(3, 3),
        )
        file.create_dataset(
            "multi/directions",
            data=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        )
        file.create_dataset(
            "multi/energies",
            data=[100.0, 200.0, 300.0],
        )

        file.create_dataset(
            "fitqun/e_1rnll",
            data=[10.0, 5.0, 2.0],
        )
        file.create_dataset(
            "fitqun/mu_1rnll",
            data=[2.0, 10.0, 8.0],
        )
        file.create_dataset(
            "fitqun/e_1rmom",
            data=[10.0, 10.0, 10.0],
        )

        file.create_dataset(
            "fitqun/e_position",
            data=np.ones((3, 3)),
        )
        file.create_dataset(
            "fitqun/e_direction",
            data=np.eye(3),
        )
        file.create_dataset(
            "fitqun/e_momentum",
            data=[50.0, 100.0, 150.0],
        )

    return path


def test_load_fitqun(fitqun_path) -> None:
    data = load_fitqun(fitqun_path)

    assert data.event_count == 3
    np.testing.assert_array_equal(
        data.labels,
        [0, 1, 1],
    )
    assert data.truth_positions.shape == (3, 3)
    assert data.truth_directions.shape == (3, 3)
    np.testing.assert_allclose(
        data.truth_energies,
        [100.0, 200.0, 300.0],
    )


def test_fitqun_electron_muon_score(fitqun_path) -> None:
    data = load_fitqun(fitqun_path)

    np.testing.assert_allclose(
        data.electron_muon_score,
        [-10.0, 3.0, 4.0],
    )

    np.testing.assert_array_equal(
        data.electron_muon_predictions(),
        [0, 1, 1],
    )


def test_load_fitqun_optional_reconstruction(
    fitqun_path,
) -> None:
    keys = FitQunKeys(
        reconstructed_positions="fitqun/e_position",
        reconstructed_directions="fitqun/e_direction",
        reconstructed_momenta="fitqun/e_momentum",
    )

    data = load_fitqun(
        fitqun_path,
        keys=keys,
    )

    assert data.reconstructed_positions is not None
    assert data.reconstructed_directions is not None
    assert data.reconstructed_momenta is not None

    assert data.reconstructed_positions.shape == (3, 3)
    assert data.reconstructed_directions.shape == (3, 3)

    np.testing.assert_allclose(
        data.reconstructed_momenta,
        [50.0, 100.0, 150.0],
    )


def test_load_fitqun_rejects_length_mismatch(
    fitqun_path,
) -> None:
    with h5py.File(fitqun_path, "a") as file:
        del file["fitqun/e_1rmom"]
        file.create_dataset(
            "fitqun/e_1rmom",
            data=[10.0, 20.0],
        )

    with pytest.raises(
        ValueError,
        match="Array lengths do not match",
    ):
        load_fitqun(fitqun_path)
