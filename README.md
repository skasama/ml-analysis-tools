# ML Analysis Tools

Analysis tools for machine-learning-based event reconstruction studies.

## Installation

Clone the repository and install the dependencies with `uv`.

```bash
git clone https://github.com/skasama/ml-analysis-tools.git
cd ml-analysis-tools
uv sync
```

Run the checks:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Examples

Examples are divided into two groups:

```text
examples/
├── simple/    Short examples for starting a new analysis
└── full/      Complete analysis workflows
```

### Simple examples

Copy a short example and edit the paths and settings near the top of the file.

```bash
cp examples/simple/plot_vertex_errors.py my_vertex_analysis.py
uv run python my_vertex_analysis.py
```

Example settings:

```python
from pathlib import Path

PREDICTIONS_PATH = Path("path/to/outputs/preds.npy")
TARGETS_PATH = Path("path/to/outputs/targets.npy")
OUTPUT_PATH = Path("outputs/vertex_errors.png")
```

### Full examples

Full examples include complete plots and statistical summaries.

```text
examples/full/
├── plot_vertex_regression.py
├── plot_direction_regression.py
├── plot_energy_regression.py
├── plot_classification.py
├── plot_resolution_dependencies.py
└── plot_classification_dependencies.py
```

Run a full example:

```bash
uv run python examples/full/plot_vertex_regression.py
```

Each example contains editable settings for:

- input and output paths
- normalization ranges
- detector geometry
- binning
- colors and line styles
- displayed statistics
- optional fiTQun comparison

## Python API

### Load NumPy arrays

```python
from analysis_tools.io import load_array_pair

predictions, targets = load_array_pair(
    "path/to/preds.npy",
    "path/to/targets.npy",
)
```

### Apply inverse normalization

```python
from analysis_tools.transforms import MinMaxTransform

transform = MinMaxTransform(
    minimum=[-1690.0, -1690.0, -1810.0],
    maximum=[1690.0, 1690.0, 1810.0],
)

predictions_cm = transform.inverse(predictions)
targets_cm = transform.inverse(targets)
```

Values outside the normalization range are not clipped.

### Calculate vertex errors

```python
from analysis_tools.metrics import vertex_errors

errors = vertex_errors(
    predictions_cm,
    targets_cm,
)
```

### Create a plot

```python
from analysis_tools.plotting import (
    plot_vertex_errors,
    save_figure,
)

figure, axis = plot_vertex_errors(
    errors,
    bins=100,
    xlabel="3D vertex error [cm]",
    color="tab:blue",
)

axis.grid(alpha=0.3)

save_figure(
    figure,
    "outputs/vertex_errors.png",
    close=True,
)
```

### Add another result

An existing Matplotlib `Axes` can be passed to a plotting function.

```python
plot_vertex_errors(
    other_errors,
    ax=axis,
    bins=100,
    label="Other method",
    color="tab:orange",
)

axis.legend()
```

## Package structure

```text
src/analysis_tools/
├── analysis/       Combined analysis results
├── data/           Array handling and event alignment
├── io/             NumPy, CSV, HDF5, YAML, and fiTQun I/O
├── metrics/        Regression and classification metrics
├── plotting/       Reusable Matplotlib functions
└── transforms/     Normalization and detector geometry
```

## Design principles

- Separate I/O, calculations, and plotting.
- Use NumPy arrays as the primary interface.
- Keep reusable calculations in `src/analysis_tools/`.
- Keep paths, layouts, and study-specific settings in `examples/`.
- Allow multiple results to be added to the same Matplotlib axes.
- Add abstractions only when duplicated functionality is identified.

## Configuration files

YAML configuration files are available for reproducible workflows.

```text
configs/
├── samples/
├── loss/
└── vertex/
```

Sample-wide settings can be reused and overridden by an individual analysis configuration.

For quick exploratory work, the direct-path examples are usually simpler.

## Development

When adding a new analysis:

1. Start from an example in `examples/simple/`.
2. Reuse functions from `analysis_tools`.
3. Keep file paths and plot-specific settings in the example.
4. Move reusable calculations into `src/analysis_tools/`.
5. Add unit tests for reusable functionality.

Before committing:

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
```

This README.md is written by Copilot to summarize development policy.