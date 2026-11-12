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

The command lists NPY, NPZ, CSV, HDF5, and HDF5-compatible `.hy` arrays with
their shapes and dtypes. NPZ, HDF5, and `.hy` variables can then be loaded with
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

The comparison command supports independent event samples for regression,
energy, vertex, direction, and classification:

```bash
analysis compare --task vertex \
  --method ml --prediction ml/preds.npy --target ml/targets.npy \
  --method other --prediction other/result.hy \
  --prediction-variable vertex_prediction \
  --target other/result.hy --target-variable vertex_truth \
  --metric x.mae \
  --output-dir plots/vertex-comparison
```

For `vertex` and `direction`, nested metrics are selected with dotted names
such as `x.mae` or `opening_angle.resolution_68`. The comparison command does
not require event IDs to match; each prediction/truth pair is evaluated in its
own stored order. Existing dependency commands remain available for
true-Wall/ToWall/energy dependent plots.

To compare complete vertex/direction/energy dependency analyses, use the
dedicated command below. Wall and ToWall are calculated from the vertex and
direction truth arrays; no `wall.npy` or `towall.npy` file is required:

```bash
analysis compare-resolution-dependencies \
  --label ml \
  --vertex-run /path/to/ml_vertex_run \
  --direction-run /path/to/ml_direction_run \
  --energy-run /path/to/ml_energy_run \
  --label fitqun \
  --vertex-run /path/to/fitqun_vertex_run \
  --direction-run /path/to/fitqun_direction_run \
  --energy-run /path/to/fitqun_energy_run \
  --output-dir plots/resolution_dependencies_comparison
```

The command overlays vertex, angular, absolute-energy, and relative-energy
resolution versus true Wall, true ToWall, and true energy. Indices are
optional: when supplied, each label's vertex/direction/energy runs are aligned
using their indices; when omitted, stored order is used and event counts must
still agree. Different labels may use different event samples.

The same command is available as:

```bash
python scripts/compare_resolution_dependencies.py ...
```

When the first group uses the default NPY layout and only a later group uses
variable-based input, a single `--*-variable` option applies to the last label.
This allows the ML group to omit variable names while the `others` group
specifies variables from a `.hy` file.

To overlay dependency profiles from multiple methods, use the dedicated
command. Regression produces bias and 68% resolution panels:

```bash
analysis compare-dependencies --task regression \
  --dependency wall \
  --method ml \
  --prediction ml/preds.npy --target ml/targets.npy \
  --method other \
  --prediction other/result.hy --prediction-variable prediction \
  --target other/result.hy --target-variable truth \
  --x-file common/dependencies.npz --x-variable wall \
  --bins 20 \
  --output-dir plots/wall-comparison
```

For classification, the output contains efficiency, purity, and mis-PID
profiles. `--x-file` may be specified once for a common dependency array or
once per method for independent samples. The same applies to the prediction,
truth, and dependency variables.

All regression plotting commands accept variable-based input files through
`--predictions-variable`, `--targets-variable`, and `--indices-variable`.
This works for NPZ, HDF5, and HDF5-compatible `.hy` files while preserving the
existing NPY defaults.

Normalized regression values can be mapped from any source interval. For
example, values stored in `[-1, 1]` can be converted to physical energy with:

```bash
analysis energy \
  --run-dir /path/to/run \
  --normalized-minimum -1 \
  --normalized-maximum 1 \
  --minimum 0.5 \
  --maximum 2000
```

The same `--normalized-minimum` and `--normalized-maximum` options are
available for vertex and direction plotting. The default remains `[0, 1]`.

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

If a result file stores multiple HDF5-compatible variables, including a file
with the `.hy` extension, specify the variable name separately for prediction
and truth. The same file can be used for both:

```bash
analysis compare --task regression \
  --method ml \
  --prediction /path/to/ml/outputs/preds.npy \
  --target /path/to/ml/outputs/targets.npy \
  --method other \
  --prediction /path/to/other/result.hy \
  --prediction-variable prediction_variable_name \
  --target /path/to/other/result.hy \
  --target-variable truth_variable_name \
  --output-dir plots/comparison
```

For a mix of NPY inputs and variable-based inputs, omit the variable option
for the NPY result; a single `--prediction-variable` or `--target-variable`
is applied to the last method:

```bash
analysis compare --task regression \
  --method ml --prediction ml/preds.npy --target ml/targets.npy \
  --method other --prediction other/result.hy \
  --prediction-variable prediction_variable_name \
  --target other/result.hy --target-variable truth_variable_name
```

Index files are optional. If an index file is supplied for every method, common
event indices are used to reorder all methods before comparison:

```bash
analysis compare --task regression \
  --method ml --prediction ml/preds.npy --target ml/targets.npy \
  --indices ml/indices.npy \
  --method other --prediction other/preds.npy --target other/targets.npy \
  --indices other/indices.npy
```

Index values do not need to have the same original order. If no index files
are supplied, each prediction/truth pair is evaluated independently in its
stored order. For NPZ/HDF5/`.hy` index arrays, also pass
`--indices-variable`.

Regression writes the selected metric comparison, a combined bias/68%
resolution comparison, and an overlaid residual histogram.
Classification writes the selected accuracy comparison and
`classification_comparison.png`, containing class-wise efficiency, purity,
and mis-PID rate. Here, mis-PID is the one-vs-rest false-positive rate
`FP/(FP+TN)`. The lower-level `compare_methods(..., align=True)` API remains
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
