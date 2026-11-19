from unittest.mock import patch

import matplotlib.pyplot as plt
import pytest

from analysis_tools.plotting.output import save_figure


def test_save_figure(tmp_path) -> None:
    figure, axis = plt.subplots()
    axis.plot([0.0, 1.0], [1.0, 2.0])

    path = tmp_path / "figures" / "plot.png"
    returned_path = save_figure(figure, path)

    assert returned_path == path
    assert path.is_file()

    plt.close(figure)


def test_save_figure_closes_figure(tmp_path) -> None:
    figure, _ = plt.subplots()
    figure_number = figure.number

    save_figure(
        figure,
        tmp_path / "plot.png",
        close=True,
    )

    assert not plt.fignum_exists(figure_number)


def test_save_figure_rejects_missing_parent(tmp_path) -> None:
    figure, _ = plt.subplots()

    with pytest.raises(FileNotFoundError):
        save_figure(
            figure,
            tmp_path / "missing" / "plot.png",
            create_parents=False,
        )

    plt.close(figure)


def test_save_figure_forwards_kwargs(
    tmp_path,
) -> None:
    figure, _ = plt.subplots()
    path = tmp_path / "plot.png"

    with patch.object(
        figure,
        "savefig",
        wraps=figure.savefig,
    ) as savefig:
        save_figure(
            figure,
            path,
            dpi=150,
            facecolor="white",
            edgecolor="black",
            metadata={
                "Creator": "ml-analysis-tools",
            },
        )

    savefig.assert_called_once()

    keyword_arguments = savefig.call_args.kwargs

    assert keyword_arguments["dpi"] == 150
    assert keyword_arguments["facecolor"] == "white"
    assert keyword_arguments["edgecolor"] == "black"
    assert keyword_arguments["metadata"] == {
        "Creator": "ml-analysis-tools",
    }
    assert keyword_arguments["bbox_inches"] == "tight"

    assert path.is_file()

    plt.close(figure)
