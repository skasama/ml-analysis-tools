"""Plot full multiclass-classification performance.

Edit the settings below and run:

    uv run python examples/full/plot_classification.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis_tools.data import as_integer_array
from analysis_tools.io import load_array_pair
from analysis_tools.metrics import (
    class_metrics,
    classification_accuracy,
    confusion_matrix,
    predicted_labels,
    roc_auc,
    roc_curve,
    softmax,
    threshold_metrics,
)
from analysis_tools.plotting import (
    plot_confusion_matrix,
    save_figure,
)
from analysis_tools.plotting.common import (
    plot_histogram,
    plot_series,
)

# ============================================================
# Input and output
# ============================================================

SCORES_PATH = Path("path/to/outputs/preds.npy")
TARGETS_PATH = Path("path/to/outputs/targets.npy")

OUTPUT_DIRECTORY = Path("outputs/classification")


# ============================================================
# Class definitions
# ============================================================

# The order must match the output columns in preds.npy.
CLASS_LABELS = np.array([0, 1, 2], dtype=np.int64)

CLASS_NAMES = (
    "Electron",
    "Muon",
    "Gamma",
)

CLASS_COLORS = (
    "tab:blue",
    "tab:orange",
    "tab:green",
)


# ============================================================
# Input interpretation
# ============================================================

# True if preds.npy contains logits.
# False if it already contains probabilities.
APPLY_SOFTMAX = True


# ============================================================
# Analysis settings
# ============================================================

N_SCORE_BINS = 50
N_THRESHOLDS = 101

THRESHOLDS = np.linspace(
    0.0,
    1.0,
    N_THRESHOLDS,
)


# ============================================================
# Plot settings
# ============================================================

DPI = 200
SHOW_FIGURES = True

HISTOGRAM_LINEWIDTH = 1.5
REFERENCE_COLOR = "black"


# ============================================================
# Output filenames
# ============================================================

CONFUSION_FILENAME = "classification_confusion_matrix.png"
SCORE_FILENAME = "classification_scores.png"
ROC_FILENAME = "classification_roc.png"
THRESHOLD_FILENAME = "classification_thresholds.png"


def prepare_classification_data() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Load scores and targets and return probabilities and labels."""
    scores, targets = load_array_pair(
        SCORES_PATH,
        TARGETS_PATH,
        first_name="scores",
        second_name="targets",
    )

    scores = np.asarray(scores, dtype=float)
    targets = as_integer_array(
        targets,
        name="targets",
    )

    if scores.ndim != 2:
        raise ValueError(f"scores must have shape (N, C), got {scores.shape}.")

    if scores.shape[1] != len(CLASS_LABELS):
        raise ValueError(
            "The number of score columns does not match "
            f"CLASS_LABELS: {scores.shape[1]} != "
            f"{len(CLASS_LABELS)}."
        )

    finite_mask = np.all(
        np.isfinite(scores),
        axis=1,
    )

    removed_event_count = int(np.count_nonzero(~finite_mask))

    scores = scores[finite_mask]
    targets = targets[finite_mask]

    if len(scores) == 0:
        raise ValueError("No finite classification events remain.")

    unknown_targets = np.setdiff1d(
        np.unique(targets),
        CLASS_LABELS,
    )

    if unknown_targets.size:
        raise ValueError(
            "Targets contain labels not listed in CLASS_LABELS: "
            f"{unknown_targets.tolist()}"
        )

    probabilities = softmax(scores) if APPLY_SOFTMAX else scores

    if not APPLY_SOFTMAX:
        probability_sums = np.sum(
            probabilities,
            axis=1,
        )

        if np.any(probabilities < 0.0):
            raise ValueError("Probabilities must be non-negative.")

        if not np.allclose(
            probability_sums,
            1.0,
        ):
            raise ValueError("Probability rows must sum to one.")

    prediction_indices = predicted_labels(probabilities)
    predictions = CLASS_LABELS[prediction_indices]

    print(f"Removed non-finite events: {removed_event_count:,}")

    return probabilities, predictions, targets


def print_statistics(
    probabilities: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
) -> None:
    """Print overall and per-class classification statistics."""
    accuracy = classification_accuracy(
        predictions,
        targets,
    )

    labels, efficiencies, purities = class_metrics(
        predictions,
        targets,
        class_labels=CLASS_LABELS,
    )

    print("\n========== Classification statistics ==========")
    print(f"Number of events: {len(targets):,}")
    print(f"Accuracy: {100.0 * accuracy:.3f}%")

    print("\nPer-class performance:")

    for column, label in enumerate(labels):
        class_index = int(np.flatnonzero(CLASS_LABELS == label)[0])
        class_name = CLASS_NAMES[class_index]

        auc_value = roc_auc(
            probabilities[:, class_index],
            targets,
            positive_label=int(label),
        )

        efficiency = efficiencies[column]
        purity = purities[column]

        efficiency_text = (
            f"{100.0 * efficiency:.3f}%" if np.isfinite(efficiency) else "undefined"
        )
        purity_text = f"{100.0 * purity:.3f}%" if np.isfinite(purity) else "undefined"

        print(f"  {class_name} ({label}):")
        print(f"    Efficiency: {efficiency_text}")
        print(f"    Purity: {purity_text}")
        print(f"    ROC AUC: {auc_value:.6f}")

    print("===============================================\n")


def plot_confusion_matrices(
    predictions: np.ndarray,
    targets: np.ndarray,
):
    """Plot count and truth-normalized confusion matrices."""
    matrix, labels = confusion_matrix(
        predictions,
        targets,
        class_labels=CLASS_LABELS,
    )

    display_names = [
        CLASS_NAMES[int(np.flatnonzero(CLASS_LABELS == label)[0])] for label in labels
    ]

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.0),
    )

    plot_confusion_matrix(
        matrix,
        display_names,
        ax=axes[0],
        normalize=None,
        title="Confusion matrix: event counts",
        colorbar=True,
        annotate=True,
    )

    plot_confusion_matrix(
        matrix,
        display_names,
        ax=axes[1],
        normalize="true",
        title="Confusion matrix: normalized by truth",
        colorbar=True,
        annotate=True,
        value_format=".3f",
    )

    figure.suptitle(
        "Multiclass classification performance",
        fontsize=16,
    )
    figure.tight_layout()

    return figure, axes


def plot_score_distributions(
    probabilities: np.ndarray,
    targets: np.ndarray,
):
    """Plot each output score separated by true class."""
    figure, axes = plt.subplots(
        1,
        len(CLASS_LABELS),
        figsize=(6.0 * len(CLASS_LABELS), 5.0),
        squeeze=False,
    )
    axes = axes[0]

    common_bins = np.linspace(
        0.0,
        1.0,
        N_SCORE_BINS + 1,
    )

    for score_index, (
        axis,
        score_name,
    ) in enumerate(
        zip(
            axes,
            CLASS_NAMES,
            strict=True,
        )
    ):
        for true_label, true_name, color in zip(
            CLASS_LABELS,
            CLASS_NAMES,
            CLASS_COLORS,
            strict=True,
        ):
            selected = targets == true_label

            plot_histogram(
                probabilities[
                    selected,
                    score_index,
                ],
                ax=axis,
                bins=common_bins,
                xlabel=f"{score_name} probability",
                ylabel="Density",
                label=f"True {true_name}",
                color=color,
                histtype="step",
                linewidth=HISTOGRAM_LINEWIDTH,
                density=True,
            )

        axis.set_xlim(0.0, 1.0)
        axis.set_title(f"{score_name} output")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)

    figure.suptitle(
        "Classification-score distributions",
        fontsize=16,
    )
    figure.tight_layout()

    return figure, axes


def plot_roc_curves(
    probabilities: np.ndarray,
    targets: np.ndarray,
):
    """Plot one-vs-rest ROC curves."""
    figure, axis = plt.subplots(figsize=(7.0, 6.0))

    for class_index, (
        class_label,
        class_name,
        color,
    ) in enumerate(
        zip(
            CLASS_LABELS,
            CLASS_NAMES,
            CLASS_COLORS,
            strict=True,
        )
    ):
        false_positive_rate, true_positive_rate, _ = roc_curve(
            probabilities[:, class_index],
            targets,
            positive_label=int(class_label),
        )

        auc_value = roc_auc(
            probabilities[:, class_index],
            targets,
            positive_label=int(class_label),
        )

        plot_series(
            false_positive_rate,
            true_positive_rate,
            ax=axis,
            xlabel="False-positive rate",
            ylabel="True-positive rate",
            label=f"{class_name} (AUC = {auc_value:.4f})",
            color=color,
            marker=None,
            linestyle="-",
            linewidth=2.0,
        )

    axis.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color=REFERENCE_COLOR,
        linestyle="--",
        linewidth=1.2,
        label="Random classifier",
    )

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_title("One-vs-rest ROC curves")
    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()

    return figure, axis


def plot_threshold_performance(
    probabilities: np.ndarray,
    targets: np.ndarray,
):
    """Plot efficiency and purity versus score threshold."""
    figure, axes = plt.subplots(
        1,
        len(CLASS_LABELS),
        figsize=(6.0 * len(CLASS_LABELS), 5.0),
        squeeze=False,
    )
    axes = axes[0]

    for class_index, (
        axis,
        class_label,
        class_name,
        color,
    ) in enumerate(
        zip(
            axes,
            CLASS_LABELS,
            CLASS_NAMES,
            CLASS_COLORS,
            strict=True,
        )
    ):
        thresholds, efficiency, purity = threshold_metrics(
            probabilities[:, class_index],
            targets,
            THRESHOLDS,
            positive_label=int(class_label),
        )

        plot_series(
            thresholds,
            efficiency,
            ax=axis,
            xlabel=f"{class_name} score threshold",
            ylabel="Rate",
            label="Efficiency",
            color=color,
            marker=None,
            linestyle="-",
            linewidth=2.0,
        )

        plot_series(
            thresholds,
            purity,
            ax=axis,
            label="Purity",
            color="black",
            marker=None,
            linestyle="--",
            linewidth=2.0,
        )

        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.05)
        axis.set_title(class_name)
        axis.grid(alpha=0.3)
        axis.legend()

    figure.suptitle(
        "Efficiency and purity versus score threshold",
        fontsize=16,
    )
    figure.tight_layout()

    return figure, axes


def main() -> None:
    probabilities, predictions, targets = prepare_classification_data()

    print_statistics(
        probabilities,
        predictions,
        targets,
    )

    confusion_figure, _ = plot_confusion_matrices(
        predictions,
        targets,
    )
    score_figure, _ = plot_score_distributions(
        probabilities,
        targets,
    )
    roc_figure, _ = plot_roc_curves(
        probabilities,
        targets,
    )
    threshold_figure, _ = plot_threshold_performance(
        probabilities,
        targets,
    )

    output_figures = (
        (
            confusion_figure,
            OUTPUT_DIRECTORY / CONFUSION_FILENAME,
        ),
        (
            score_figure,
            OUTPUT_DIRECTORY / SCORE_FILENAME,
        ),
        (
            roc_figure,
            OUTPUT_DIRECTORY / ROC_FILENAME,
        ),
        (
            threshold_figure,
            OUTPUT_DIRECTORY / THRESHOLD_FILENAME,
        ),
    )

    for figure, path in output_figures:
        save_figure(
            figure,
            path,
            dpi=DPI,
            close=not SHOW_FIGURES,
        )
        print(f"Saved: {path}")

    if SHOW_FIGURES:
        plt.show()


if __name__ == "__main__":
    main()
