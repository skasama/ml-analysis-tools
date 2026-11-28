"""Compare vertex reconstruction samples from ML runs and fiTQun.

Edit the paths, HDF5 keys, class label, and detector geometry below to match
your files. Predictions and truth positions must use the same physical units.
The default comparison is independent-sample; optional matching uses explicit
file/event ID keys and does not subtract one reconstruction from the other.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from analysis_tools import (
    EventIndex,
    EventSelection,
    MLRunConfig,
    calculate_vertex_quantities,
    evaluate_dependency,
    load_ml_run,
    match_event_indices,
    plot_evaluation,
    save_figure,
)
from analysis_tools.data import CutFlowEntry, as_integer_array
from analysis_tools.io import load_hdf5
from analysis_tools.metrics import standard_error
from analysis_tools.quantities import VertexQuantities
from analysis_tools.transforms import MinMaxTransform
from analysis_tools.transforms.geometry import compute_wall

# ============================================================
# Study-specific settings
# ============================================================


@dataclass(frozen=True)
class FitQunConfig:
    """Input and display settings for one fiTQun sample."""

    name: str
    path: Path
    color: str


ML_RUNS = [
    MLRunConfig(
        name="ML model A",
        source_hdf5=Path("data/test.h5"),
        predictions_path=Path("outputs/model_a/preds.npy"),
        targets_path=Path("outputs/model_a/targets.npy"),
        indices_path=Path("outputs/model_a/indices.npy"),
        color="tab:blue",
    ),
    MLRunConfig(
        name="ML model B",
        source_hdf5=Path("data/test.h5"),
        predictions_path=Path("outputs/model_b/preds.npy"),
        targets_path=Path("outputs/model_b/targets.npy"),
        indices_path=Path("outputs/model_b/indices.npy"),
        color="tab:orange",
    ),
]


FITQUN_CONFIGS = [
    FitQunConfig(
        name="fiTQun electron fit",
        path=Path("data/common_multi_fitqun.hy"),
        color="black",
    ),
]

FITQUN_LABEL_KEY = "multi/labels"
FITQUN_TRUTH_POSITION_KEY = "multi/positions"
FITQUN_RECON_POSITION_KEY = "fitqun/e_1rpos"
FITQUN_NHIT_KEY = "fitqun/nhit"
ML_NHIT_KEY = "nhit"

# Optional event matching. Keep disabled for independent-sample comparisons.
# Configure equivalent, unique file-ID and event-ID keys in each HDF5 file
# before enabling. If root_files need conversion to fiTQun file IDs, perform
# that mapping explicitly in this script before constructing EventIndex.
ENABLE_MATCHED_COMPARISON = False
ML_FILE_ID_KEY = "matching/file_ids"
ML_EVENT_ID_KEY = "matching/multi_event_ids"
FITQUN_FILE_ID_KEY = "matching/file_ids"
FITQUN_EVENT_ID_KEY = "matching/multi_event_ids"

SELECTED_LABEL = 1
MINIMUM_NHIT: int | None = None
# Apply one identical fiducial cut to every sample only when the comparison
# specifically targets this detector region.
APPLY_WALL_CUT = False
MINIMUM_WALL_CM = 100.0
MAXIMUM_WALL_CM = 1650.0
DETECTOR_RADIUS_CM = 1690.0
DETECTOR_HALF_HEIGHT_CM = 1810.0
WALL_EDGES_CM = np.arange(0.0, 1701.0, 100.0)
MINIMUM_EVENTS_PER_BIN = 10
OUTPUT_DIRECTORY = Path("outputs/vertex_comparison")

ML_PREDICTIONS_NORMALIZED = False
VERTEX_MINIMUM_CM = [-1690.0, -1690.0, -1810.0]
VERTEX_MAXIMUM_CM = [1690.0, 1690.0, 1810.0]


@dataclass(frozen=True)
class VertexSample:
    """Selected event-level data for one reconstruction sample."""

    name: str
    color: str
    quantities: VertexQuantities
    wall: NDArray[np.float64]
    cut_flow: tuple[CutFlowEntry, ...] = ()
    event_index: EventIndex | None = None


def _build_selection(
    labels: NDArray,
    wall: NDArray[np.float64],
    nhit: NDArray | None = None,
) -> EventSelection:
    """Apply the same truth-label, optional nhit, and Wall cuts to all samples."""
    labels = as_integer_array(labels, name="labels")
    selection = EventSelection(len(labels))
    selection.add(
        f"label == {SELECTED_LABEL}",
        labels == SELECTED_LABEL,
    )
    if MINIMUM_NHIT is not None:
        if nhit is None:
            raise ValueError(
                "A nhit dataset is required when MINIMUM_NHIT is configured."
            )
        nhit = as_integer_array(nhit, name="nhit")
        selection.add(
            f"nhit >= {MINIMUM_NHIT}",
            nhit >= MINIMUM_NHIT,
        )
    if APPLY_WALL_CUT:
        selection.add(
            f"{MINIMUM_WALL_CM:g} <= Wall <= {MAXIMUM_WALL_CM:g} cm",
            (wall >= MINIMUM_WALL_CM) & (wall <= MAXIMUM_WALL_CM),
        )
    return selection


def _make_sample(
    *,
    name: str,
    color: str,
    predictions: NDArray,
    targets: NDArray,
    labels: NDArray,
    nhit: NDArray | None = None,
    event_index: EventIndex | None = None,
) -> VertexSample:
    """Calculate quantities and apply event-level cuts consistently."""
    quantities = calculate_vertex_quantities(predictions, targets)
    wall = compute_wall(
        targets,
        radius=DETECTOR_RADIUS_CM,
        half_height=DETECTOR_HALF_HEIGHT_CM,
    )
    selection = _build_selection(
        labels,
        wall,
        nhit=nhit,
    )
    mask = selection.mask

    selected_index = None
    if event_index is not None:
        selected_index = EventIndex.from_arrays(
            event_index.file_ids[mask],
            event_index.event_ids[mask],
        )

    return VertexSample(
        name=name,
        color=color,
        quantities=quantities.select(mask),
        wall=wall[mask],
        cut_flow=selection.cut_flow,
        event_index=selected_index,
    )


def load_samples() -> tuple[list[VertexSample], list[VertexSample]]:
    """Load multiple ML runs and fiTQun samples."""
    ml_keys = [
        "multi/labels",
        "multi/positions",
    ]
    if MINIMUM_NHIT is not None:
        ml_keys.append(ML_NHIT_KEY)
    fitqun_keys = [
        FITQUN_LABEL_KEY,
        FITQUN_TRUTH_POSITION_KEY,
        FITQUN_RECON_POSITION_KEY,
    ]
    if MINIMUM_NHIT is not None:
        fitqun_keys.append(FITQUN_NHIT_KEY)

    if ENABLE_MATCHED_COMPARISON:
        ml_keys.extend([ML_FILE_ID_KEY, ML_EVENT_ID_KEY])
        fitqun_keys.extend([FITQUN_FILE_ID_KEY, FITQUN_EVENT_ID_KEY])

    ml_samples = []
    for config in ML_RUNS:
        run = load_ml_run(
            config,
            hdf5_keys=ml_keys,
        )
        event_index = None

        if ENABLE_MATCHED_COMPARISON:
            event_index = EventIndex.from_arrays(
                run.source[ML_FILE_ID_KEY],
                run.source[ML_EVENT_ID_KEY],
            )

        predictions = run.predictions.predictions
        if ML_PREDICTIONS_NORMALIZED:
            transform = MinMaxTransform(
                minimum=VERTEX_MINIMUM_CM,
                maximum=VERTEX_MAXIMUM_CM,
            )
            predictions = transform.inverse(predictions)

        ml_samples.append(
            _make_sample(
                name=run.name,
                color=run.color or "tab:blue",
                predictions=predictions,
                targets=run.source["multi/positions"],
                labels=run.source["multi/labels"],
                nhit=run.source.get(ML_NHIT_KEY),
                event_index=event_index,
            )
        )

    fitqun_samples = []
    for config in FITQUN_CONFIGS:
        fitqun_data = load_hdf5(
            config.path,
            keys=fitqun_keys,
        )

        fitqun_index = None
        if ENABLE_MATCHED_COMPARISON:
            fitqun_index = EventIndex.from_arrays(
                fitqun_data[FITQUN_FILE_ID_KEY],
                fitqun_data[FITQUN_EVENT_ID_KEY],
            )

        fitqun_samples.append(
            _make_sample(
                name=config.name,
                color=config.color,
                predictions=fitqun_data[FITQUN_RECON_POSITION_KEY],
                targets=fitqun_data[FITQUN_TRUTH_POSITION_KEY],
                labels=fitqun_data[FITQUN_LABEL_KEY],
                nhit=fitqun_data.get(FITQUN_NHIT_KEY),
                event_index=fitqun_index,
            )
        )

    return ml_samples, fitqun_samples


def plot_samples(
    samples: list[VertexSample],
    *,
    output_stem: str,
) -> None:
    """Plot shared-bin errors, resolution, bias, and event counts."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    if not samples:
        raise ValueError("At least one sample is required for plotting.")

    pooled_errors = np.concatenate(
        [
            sample.quantities.errors_3d[np.isfinite(sample.quantities.errors_3d)]
            for sample in samples
        ]
    )
    if len(pooled_errors) == 0:
        raise ValueError("No finite selected 3D errors are available to plot.")
    error_bins = np.histogram_bin_edges(pooled_errors, bins="auto")

    figure, axis = plt.subplots(figsize=(7.0, 5.0))
    for sample in samples:
        axis.hist(
            sample.quantities.errors_3d[np.isfinite(sample.quantities.errors_3d)],
            bins=error_bins,
            density=True,
            histtype="step",
            linewidth=1.5,
            label=f"{sample.name} (N={sample.quantities.event_count:,})",
            color=sample.color,
        )
    axis.set(
        xlabel="3D vertex error [cm]",
        ylabel="Probability density",
        title="Selected-sample error distributions",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    save_figure(
        figure,
        OUTPUT_DIRECTORY / f"{output_stem}_error_distributions.png",
        close=True,
    )

    resolution_quantities = {
        "X": (lambda sample: sample.quantities.residual_x, _half_central_68_width),
        "Y": (lambda sample: sample.quantities.residual_y, _half_central_68_width),
        "Z": (lambda sample: sample.quantities.residual_z, _half_central_68_width),
        "R": (lambda sample: sample.quantities.residual_r, _half_central_68_width),
        "3D": (lambda sample: sample.quantities.errors_3d, _central_68_magnitude),
    }
    resolution_figure, resolution_axes = plt.subplots(
        3,
        2,
        figsize=(12.0, 11.0),
        sharex=True,
    )
    resolution_results = []
    for sample in samples:
        sample_results = {}
        for axis, (quantity_name, (getter, metric)) in zip(
            resolution_axes.flat[:5],
            resolution_quantities.items(),
            strict=True,
        ):
            result = evaluate_dependency(
                sample.wall,
                getter(sample),
                WALL_EDGES_CM,
                metric=metric,
                minimum_events=MINIMUM_EVENTS_PER_BIN,
            )
            sample_results[quantity_name] = result
            plot_evaluation(
                result,
                ax=axis,
                label=sample.name,
                color=sample.color,
                marker="o",
                markersize=3,
            )
        resolution_results.append(sample_results)

    for axis, quantity_name in zip(
        resolution_axes.flat[:5],
        resolution_quantities,
        strict=True,
    ):
        axis.set_title(f"{quantity_name} resolution")
        axis.set_ylabel("Resolution [cm]")
        axis.grid(alpha=0.25)
    resolution_axes.flat[0].legend(fontsize="small")

    count_axis = resolution_axes.flat[5]
    for sample, sample_results in zip(samples, resolution_results, strict=True):
        result = sample_results["3D"]
        count_axis.step(
            result.centers,
            result.counts,
            where="mid",
            label=sample.name,
            color=sample.color,
        )
    count_axis.set_title("Selected events per Wall bin")
    count_axis.set_ylabel("Events")
    count_axis.set_xlabel("True Wall [cm]")
    count_axis.grid(alpha=0.25)
    count_axis.legend(fontsize="small")
    resolution_axes.flat[4].set_xlabel("True Wall [cm]")
    save_figure(
        resolution_figure,
        OUTPUT_DIRECTORY / f"{output_stem}_resolution.png",
        close=True,
    )

    bias_quantities = {
        "X": lambda sample: sample.quantities.residual_x,
        "Y": lambda sample: sample.quantities.residual_y,
        "Z": lambda sample: sample.quantities.residual_z,
        "R": lambda sample: sample.quantities.residual_r,
    }
    bias_figure, bias_axes = plt.subplots(
        2,
        2,
        figsize=(11.0, 8.0),
        sharex=True,
    )
    for sample in samples:
        for axis, (quantity_name, getter) in zip(
            bias_axes.flat,
            bias_quantities.items(),
            strict=True,
        ):
            result = evaluate_dependency(
                sample.wall,
                getter(sample),
                WALL_EDGES_CM,
                metric=np.mean,
                error=standard_error,
                minimum_events=MINIMUM_EVENTS_PER_BIN,
            )
            plot_evaluation(
                result,
                ax=axis,
                label=sample.name,
                color=sample.color,
                marker="o",
                markersize=3,
                capsize=2,
            )
            axis.set_title(f"{quantity_name} bias")
            axis.set_ylabel("Mean signed residual [cm]")
            axis.axhline(0.0, color="0.4", linewidth=0.8, linestyle="--")
            axis.grid(alpha=0.25)
    bias_axes.flat[0].legend(fontsize="small")
    bias_axes.flat[2].set_xlabel("True Wall [cm]")
    bias_axes.flat[3].set_xlabel("True Wall [cm]")
    save_figure(
        bias_figure,
        OUTPUT_DIRECTORY / f"{output_stem}_bias.png",
        close=True,
    )


def _half_central_68_width(values: NDArray[np.float64]) -> float:
    """Return half the central 68% interval width for signed residuals."""
    lower, upper = np.quantile(values, [0.16, 0.84])
    return float((upper - lower) / 2.0)


def _central_68_magnitude(values: NDArray[np.float64]) -> float:
    """Return the 68th percentile of non-negative 3D error magnitude."""
    return float(np.quantile(values, 0.68))


def make_matched_samples(
    ml_samples: list[VertexSample],
    fitqun_samples: list[VertexSample],
) -> list[VertexSample]:
    """Align each selected ML sample and fiTQun by explicit event IDs."""
    matched_samples = []
    for ml_sample in ml_samples:
        if ml_sample.event_index is None:
            raise ValueError(f"{ml_sample.name} has no configured event IDs.")

        for fitqun_sample in fitqun_samples:
            if fitqun_sample.event_index is None:
                raise ValueError(f"{fitqun_sample.name} has no configured event IDs.")

            ml_rows, fitqun_rows = match_event_indices(
                ml_sample.event_index,
                fitqun_sample.event_index,
            )

            matched_samples.append(
                VertexSample(
                    name=f"{ml_sample.name} (matched)",
                    color=ml_sample.color,
                    quantities=ml_sample.quantities.select(ml_rows),
                    wall=ml_sample.wall[ml_rows],
                )
            )
            matched_samples.append(
                VertexSample(
                    name=f"{fitqun_sample.name} matched to {ml_sample.name}",
                    color=fitqun_sample.color,
                    quantities=fitqun_sample.quantities.select(fitqun_rows),
                    wall=fitqun_sample.wall[fitqun_rows],
                )
            )

    return matched_samples


def main() -> None:
    ml_samples, fitqun_samples = load_samples()
    samples = [*ml_samples, *fitqun_samples]
    for sample in samples:
        print(f"\n{sample.name}")
        print(f"  Initial events: {sample.cut_flow[0].before:,}")
        for entry in sample.cut_flow:
            print(
                f"  {entry.name}: {entry.before:,} -> {entry.after:,} "
                f"(removed {entry.removed:,})"
            )

    # Independent comparison: event sets may differ. Apply the same cut and
    # metric definitions, but do not perform event matching.
    plot_samples(
        samples,
        output_stem="independent_vertex_comparison",
    )

    if ENABLE_MATCHED_COMPARISON:
        matched_samples = make_matched_samples(
            ml_samples,
            fitqun_samples,
        )
        plot_samples(
            matched_samples,
            output_stem="matched_vertex_comparison",
        )


if __name__ == "__main__":
    main()
