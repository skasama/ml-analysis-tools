import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "analysis_tools",
        "analysis_tools.io",
        "analysis_tools.data",
        "analysis_tools.transforms",
        "analysis_tools.metrics",
        "analysis_tools.plotting",
        "analysis_tools.analysis",
        "analysis_tools.physics",
        "analysis_tools.splitting",
    ],
)
def test_module_can_be_imported(module_name: str) -> None:
    module = importlib.import_module(module_name)

    assert module is not None
