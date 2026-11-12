# Analysis Tools

This repository provides analysis tools that use explicitly specified input files instead of automatically searching for them.

Relative input paths are resolved from the directory specified by `--run-dir`.

## Command-line interface

After installing the package, use the unified `analysis` command. The
task-specific options remain unchanged:

```bash
analysis energy \
  --run-dir /path/to/run \
  --output-dir plots/energy
```

Available commands are `classification`, `classification-dependencies`,
`direction`, `energy`, `loss`, `resolution-dependencies`, `vertex`, and
`event-display`, plus `inputs` and `compare`. Direct entry points such as
`analysis-plot-energy` are also provided for scripts and batch jobs.

To inspect available input files and variables before selecting them:

```bash
analysis inputs --run-dir /path/to/run
```

The command lists NPY, NPZ, CSV, and HDF5 arrays with their shapes and dtypes.
NPZ and HDF5 variables can then be loaded with
`analysis.io.load_variable(path, variable)`.

The existing files under `scripts/` remain compatible as transitional wrappers.
New plotting commands should be added as modules under `src/analysis/cli` and
registered in `COMMANDS` and `[project.scripts]`.

Common calculations belong in `analysis.metrics`, and reusable figure
construction belongs in `analysis.plotting`. For example:

```python
from analysis.metrics import residual_profile, resolution68
from analysis.plotting import save, scatter_identity

resolution = resolution68(predictions - targets)
centres, biases, resolutions, counts = residual_profile(
    targets,
    predictions - targets,
    n_bins=20,
)
figure = scatter_identity(
    predictions,
    targets,
    xlabel="True energy",
    ylabel="Predicted energy",
    title="Energy regression",
    unit="MeV",
)
save(figure, "plots/energy/prediction_vs_truth.png")
```

The package also provides small building blocks for larger comparisons:
`analysis.data` contains the common result schema, `analysis.comparison`
aligns and evaluates methods, `analysis.uncertainty` provides bootstrap
intervals, `analysis.config` loads validated YAML mappings, and
`analysis.baselines` contains simple non-ML baselines. Plot-specific helpers
are grouped under `analysis.plotting.comparison` and
`analysis.plotting.diagnostics`.

## Running in Apptainer

Add the local `src` directory to `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

## Plotting Training Loss

```bash
python scripts/plot_loss.py \
  --run-dir /path/to/run \
  --train-log outputs/log_train_0.csv \
  --val-log outputs/log_val.csv \
  --output-dir plots/loss \
  --output-file loss_curve_readable.png
```

The paths specified by `--train-log` and `--val-log` are resolved relative to `--run-dir`.

In the example above, the following files are used:

```text
/path/to/run/outputs/log_train_0.csv
/path/to/run/outputs/log_val.csv
```

## Common Run Outputs

The classification, vertex-regression, and direction-regression scripts use the following input options:

- `--predictions`: prediction array
- `--targets`: target array
- `--indices`: event-index array

For example:

```bash
python scripts/plot_direction_regression.py \
  --run-dir /path/to/run \
  --predictions outputs/preds.npy \
  --targets outputs/targets.npy \
  --indices outputs/indices.npy \
  --output-dir plots/direction
```

The same options are available for:

```text
plot_classification.py
plot_vertex_regression.py
```

## Resolution Dependencies Using Multiple Runs

`plot_resolution_dependencies.py` combines the outputs of the vertex, direction, and energy regression runs.

The input files must be specified separately for each run using options such as:

- `--vertex-predictions`
- `--vertex-targets`
- `--vertex-indices`
- `--direction-predictions`
- `--direction-targets`
- `--direction-indices`
- `--energy-predictions`
- `--energy-targets`
- `--energy-indices`

Example:

```bash
python scripts/plot_resolution_dependencies.py \
  --vertex-run /path/to/vertex_run \
  --vertex-predictions outputs/preds.npy \
  --vertex-targets outputs/targets.npy \
  --vertex-indices outputs/indices.npy \
  --direction-run /path/to/direction_run \
  --direction-predictions outputs/preds.npy \
  --direction-targets outputs/targets.npy \
  --direction-indices outputs/indices.npy \
  --energy-run /path/to/energy_run \
  --energy-predictions outputs/preds.npy \
  --energy-targets outputs/targets.npy \
  --energy-indices outputs/indices.npy \
  --output-dir plots/resolution_dependencies
```

## 3D Event Display

The HDF5 dataset names and geometry-array key can also be specified explicitly.

```bash
python scripts/save_event_display_3d.py \
  --h5file events.h5 \
  --geometry-file geometry.npz \
  --event-index-dataset event_hits_index \
  --pmt-dataset hit_pmt \
  --time-dataset hit_time \
  --charge-dataset hit_charge \
  --position-key position \
  --event 0 \
  --channel charge \
  --output plots/event0.png
```

## Comparing Analysis Methods

Two or more independent regression or classification results can be compared
without requiring matching event IDs. Specify the task, method labels, files,
and optional NPZ/HDF5 variables. For regression with one shared truth:

```bash
analysis compare --task regression \
  --method ml \
  --prediction /path/to/ml/predictions.npy \
  --method baseline \
  --prediction /path/to/baseline/predictions.npy \
  --truth /path/to/common/targets.npy \
  --metric mae \
  --output-dir plots/comparison
```

For separate truth/result pairs:

```bash
analysis compare --task regression \
  --method model1 \
  --prediction /path/to/model1/predictions.npz \
  --prediction-variable predictions \
  --target /path/to/model1/targets.npz \
  --target-variable truth \
  --method model2 \
  --prediction /path/to/model2/predictions.npz \
  --prediction-variable predictions \
  --target /path/to/model2/targets.npz \
  --target-variable truth
```

Classification uses the same input form and compares accuracy:

```bash
analysis compare --task classification \
  --method model1 --prediction model1/preds.npy --target model1/targets.npy \
  --method model2 --prediction model2/preds.npy --target model2/targets.npy \
  --output-dir plots/classification-comparison
```

Regression writes a metric comparison and an overlaid residual histogram.
Classification writes the selected metric comparison. Event IDs are not
loaded or aligned by this command; each prediction/truth pair is evaluated
independently. The lower-level `compare_methods(..., align=True)` API remains
available when event-level alignment is specifically required.

Results from ML and non-ML methods can be represented with the same
`MethodResult` object. Event IDs are required so that methods can either be
checked for identical ordering or aligned on their common event set.

```python
import numpy as np

from analysis.metrics import compare_metrics, regression_metrics
from analysis.results import MethodResult, align_by_event_id

ml = MethodResult(
    method="ml",
    task="energy_regression",
    predictions=np.load("ml/predictions.npy"),
    targets=np.load("ml/targets.npy"),
    event_ids=np.load("ml/event_ids.npy"),
)
baseline = MethodResult(
    method="baseline",
    task="energy_regression",
    predictions=np.load("baseline/predictions.npy"),
    targets=np.load("baseline/targets.npy"),
    event_ids=np.load("baseline/event_ids.npy"),
)

ml, baseline = align_by_event_id(ml, baseline)
metrics = compare_metrics((ml, baseline), regression_metrics)
```

`align_results` should be used when identical event ordering is required;
`align_by_event_id` explicitly selects the common event-ID intersection and
reorders each method before calculating metrics.

`analysis.metrics` also provides `correlation` and `vertex_metrics`. The latter
reports Cartesian, cylindrical-radius, Wall, 3D error, and optional ToWall
statistics using the detector geometry. Energy and vertex regression profile
plots include bootstrap error bars for bias and resolution; the number of
bootstrap samples can be adjusted with `--profile-bootstrap-samples`.

## Development

Install the locked dependencies, including development tools, with:

```bash
uv sync --locked
```

Run the same checks used by CI:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```
