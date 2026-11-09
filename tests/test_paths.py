import pytest

from analysis.paths import input_path, output_path


def test_input_path_resolves_relative_file(tmp_path):
    source = tmp_path / "input.txt"
    source.write_text("data")

    assert input_path(tmp_path, "input.txt", "Input") == source


def test_input_path_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Input was not found"):
        input_path(tmp_path, "missing.txt", "Input")


def test_output_path_creates_parent_directory(tmp_path):
    result = output_path(tmp_path / "plots", "nested/result.png")

    assert result == tmp_path / "plots/nested/result.png"
    assert result.parent.is_dir()
