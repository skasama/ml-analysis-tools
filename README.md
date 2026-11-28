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
├── simple/       Small, focused examples
├── full/         End-to-end data workflows
└── notebooks/    Interactive versions and walkthroughs
```

### Simple examples

Use these to learn one operation or try an API with synthetic data:

```bash
uv run python examples/simple/task_api_walkthrough.py
```

`task_api_walkthrough.py` runs without input files and demonstrates event
selection with cut flow, aligned task quantities (vertex, direction, energy,
and classification), dependency evaluation, and saved plots. To start from
real `.npy` prediction files, use `load_prediction_data.py` or
`plot_vertex_errors.py`; edit the input and output paths near the top of the
script. `evaluate_results.py` is another small example of a binned metric and
uncertainty plot.

### Full workflows

| Example | What it demonstrates |
| --- | --- |
| `examples/full/plot_vertex_regression.py` | Vertex prediction, residuals, and resolution |
| `examples/full/plot_direction_regression.py` | Direction residuals, opening angles, and degree-based angular metrics |
| `examples/full/plot_energy_regression.py` | Absolute and fractional energy residuals and profiles |
| `examples/full/plot_classification.py` | Classification performance and score-based metrics |
| `examples/full/plot_resolution_dependencies.py` | Resolution versus Wall, ToWall, and energy |
| `examples/full/plot_classification_dependencies.py` | Efficiency and purity versus event dependencies |
| `examples/full/plot_resolution_dependencies_ml_vs_fitqun.py` | Multi-task resolution dependencies for ML and fiTQun |
| `examples/full/plot_classification_dependencies_ml_vs_fitqun.py` | Classification efficiency/purity comparison for ML and fiTQun |
| `examples/full/compare_vertex_samples.py` | Multiple ML/fiTQun samples, shared cuts, optional normalization and ID-matched comparison |

These are data-dependent scripts: set the file paths, dataset keys, units,
normalization ranges, and selection settings before running. The two
`*_ml_vs_fitqun.py` scripts use their own settings and fit-parameter
conventions; the `compare_vertex_samples.py` workflow instead supports multiple
input samples and optional event-ID alignment. For example:

```bash
uv run python examples/full/compare_vertex_samples.py
```

### Notebooks

Open the notebooks from the repository root:

```bash
uv run jupyter lab examples/notebooks
```

`task_api_walkthrough.ipynb` is runnable without private inputs.
`compare_vertex_samples.ipynb` follows the complete file-to-plot workflow;
edit its paths and dataset keys first.

`compare_vertex_samples.py` and its matching notebook show the full HDF5/ML
loading, event-selection, evaluation, and plotting workflow. The truth-label
selection is shared across samples; optional nhit and fiducial-Wall cuts are
configured explicitly and applied consistently. The default does not impose a
Wall cut. Independent comparisons do not assume event correspondence. Optional
matched comparisons require equivalent, unique file/event identifiers in all
inputs. Resolution and bias are shown separately: signed X/Y/Z/R resolution
uses half the central 16th–84th percentile interval, 3D resolution uses the
68th percentile of the error magnitude, and bias uses the mean signed residual
with standard error. These are explicit example metric choices and should be
adapted to the analysis definition.

All examples use the same principle: keep each cut mask aligned across every
event-level quantity, use common bin edges when comparing distributions, and
state the metric and uncertainty being plotted. The plots are illustrative;
they are not a substitute for analysis-specific validation of cuts, units, or
statistical definitions.

## Python API

### Evaluate and plot in a few lines

```python
import numpy as np

from analysis_tools import (
    calculate_vertex_quantities,
    evaluate_dependency,
    plot_evaluation,
)

quantities = calculate_vertex_quantities(predictions, targets)

evaluation = evaluate_dependency(
    wall,
    quantities.errors_3d,
    edges=[0.0, 50.0, 100.0, 150.0],
    metric=np.mean,
    minimum_events=10,
)

axis = plot_evaluation(
    evaluation,
    label="ML",
    color="tab:blue",
)
axis.grid(alpha=0.25)
```

A complete runnable example is available at
`examples/simple/evaluate_results.py`.

For a complete file-to-plot workflow comparing multiple ML runs with fiTQun,
including label selection, optional nhit/fiducial cuts, and matched-event
analysis, see:

- `examples/full/compare_vertex_samples.py`
- `examples/notebooks/compare_vertex_samples.ipynb`

For ML outputs that must stay aligned to their source HDF5 rows:

```python
from analysis_tools import MLRunConfig, load_ml_run

run = load_ml_run(
    MLRunConfig(
        name="ML",
        source_hdf5="data/source.hy",
        predictions_path="outputs/preds.npy",
        targets_path="outputs/targets.npy",
        indices_path="outputs/indices.npy",
        color="tab:blue",
    ),
    hdf5_keys=["multi/labels", "multi/positions", "multi/energies"],
)

labels = run.source["multi/labels"]
selected_run = run.select(selection.mask)
```

`hdf5_keys` remain full HDF5 paths. `run.select()` applies one sample's
Boolean or integer selection to predictions, targets, indices, and all loaded
source arrays together. If `indices_path` is omitted, source HDF5 rows must
already be in prediction order and have the same event count; only the count
can be validated. Direction quantities return theta, phi, their residuals,
and `opening_angles` in degrees by default. Phi residuals use the shortest
signed difference in [-180, 180] degrees. Pass
`calculate_direction_quantities(..., degrees=False)` for radians.

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