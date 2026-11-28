import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = (
    Path(__file__).parents[1] / "examples" / "full" / "compare_vertex_samples.py"
)
SPEC = importlib.util.spec_from_file_location("vertex_comparison_example", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
EXAMPLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXAMPLE
SPEC.loader.exec_module(EXAMPLE)


def test_resolution_statistics_use_central_68_percent_definitions():
    signed_residuals = np.arange(101, dtype=float)
    error_magnitudes = np.arange(101, dtype=float)

    assert EXAMPLE._half_central_68_width(signed_residuals) == pytest.approx(34.0)
    assert EXAMPLE._central_68_magnitude(error_magnitudes) == pytest.approx(68.0)


def test_wall_cut_is_optional_and_shared_selection_remains_explicit(monkeypatch):
    monkeypatch.setattr(EXAMPLE, "SELECTED_LABEL", 1)
    monkeypatch.setattr(EXAMPLE, "MINIMUM_NHIT", 5)
    monkeypatch.setattr(EXAMPLE, "MINIMUM_WALL_CM", 100.0)
    monkeypatch.setattr(EXAMPLE, "MAXIMUM_WALL_CM", 1650.0)
    labels = np.array([1, 1, 0, 1])
    nhit = np.array([5, 6, 8, 4])
    wall = np.array([50.0, 200.0, 300.0, 400.0])

    monkeypatch.setattr(EXAMPLE, "APPLY_WALL_CUT", False)
    without_wall_cut = EXAMPLE._build_selection(labels, wall, nhit).mask
    np.testing.assert_array_equal(without_wall_cut, [True, True, False, False])

    monkeypatch.setattr(EXAMPLE, "APPLY_WALL_CUT", True)
    with_wall_cut = EXAMPLE._build_selection(labels, wall, nhit).mask
    np.testing.assert_array_equal(with_wall_cut, [False, True, False, False])
