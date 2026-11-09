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
`event-display`. Direct entry points such as `analysis-plot-energy` are also
provided for scripts and batch jobs.

The existing files under `scripts/` remain compatible as transitional wrappers.
New plotting commands should be added as modules under `src/analysis/cli` and
registered in `COMMANDS` and `[project.scripts]`.

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
