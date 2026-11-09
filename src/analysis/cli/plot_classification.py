import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from analysis.io import load_outputs
from analysis.paths import output_path
from analysis.plotting import save

CLASS_NAMES = ["muon", "electron", "pion"]


def prepare_targets(values, n_classes):
    """Convert targets to integer class labels."""
    values = np.asarray(values)

    if values.ndim == 1:
        targets = values
    elif values.ndim == 2 and values.shape[1] == 1:
        targets = values[:, 0]
    elif values.ndim == 2 and values.shape[1] == n_classes:
        targets = np.argmax(values, axis=1)
    else:
        raise ValueError(
            "Targets must have shape (N,), (N, 1), or "
            f"(N, {n_classes}), but received {values.shape}."
        )

    targets = targets.astype(int)
    valid_labels = np.arange(n_classes)

    if not np.all(np.isin(targets, valid_labels)):
        raise ValueError(
            f"Targets contain invalid labels: {np.unique(targets)}. "
            f"Expected labels: {valid_labels.tolist()}."
        )

    return targets


def prepare_logits(values, n_classes):
    """Validate model outputs as classification logits."""
    logits = np.asarray(values, dtype=float)

    if logits.ndim != 2 or logits.shape[1] != n_classes:
        raise ValueError(
            f"Predictions must have shape (N, {n_classes}), "
            f"but received {logits.shape}."
        )

    return logits


def add_matrix_text(ax, matrix, value_format):
    """Add numerical values to a confusion-matrix axis."""
    maximum = np.max(matrix) if matrix.size else 0.0
    threshold = 0.5 * maximum

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]

            ax.text(
                column,
                row,
                format(value, value_format),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=11,
            )


def build_parser():
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Plot particle-classification confusion matrices."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/preds.npy"),
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("outputs/targets.npy"),
    )
    parser.add_argument(
        "--indices",
        type=Path,
        default=Path("outputs/indices.npy"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/classification"),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("classification_confusion_matrix.png"),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--show",
        action="store_true",
    )

    return parser


def main():
    """Plot count and row-normalized confusion matrices."""
    args = build_parser().parse_args()

    outputs = load_outputs(
        args.run_dir,
        args.predictions,
        args.targets,
        args.indices,
    )

    n_classes = outputs.predictions.shape[1]

    if n_classes > len(CLASS_NAMES):
        raise ValueError(
            f"Received {n_classes} classes, but only "
            f"{len(CLASS_NAMES)} class names are defined."
        )

    class_names = CLASS_NAMES[:n_classes]

    logits = prepare_logits(
        outputs.predictions,
        n_classes,
    )
    targets = prepare_targets(
        outputs.targets,
        n_classes,
    )

    finite_events = np.all(np.isfinite(logits), axis=1)

    if np.any(~finite_events):
        n_removed = int(np.sum(~finite_events))
        print(f"Warning: removing {n_removed} events with non-finite logits.")

        logits = logits[finite_events]
        targets = targets[finite_events]

    if len(targets) == 0:
        raise ValueError("No valid events remain.")

    predicted_classes = np.argmax(
        logits,
        axis=1,
    )

    matrix_counts = confusion_matrix(
        targets,
        predicted_classes,
        labels=np.arange(n_classes),
    )

    row_sums = matrix_counts.sum(
        axis=1,
        keepdims=True,
    )

    matrix_normalized = np.divide(
        matrix_counts,
        row_sums,
        out=np.zeros_like(
            matrix_counts,
            dtype=float,
        ),
        where=row_sums != 0,
    )

    accuracy = accuracy_score(
        targets,
        predicted_classes,
    )

    print(f"Number of events: {len(targets)}")
    print(f"Accuracy: {accuracy:.6f}")
    print("Confusion matrix:")
    print(matrix_counts)

    configurations = [
        {
            "matrix": matrix_counts,
            "title": "Event count",
            "format": "d",
            "vmax": None,
            "colorbar_label": "Events",
        },
        {
            "matrix": matrix_normalized,
            "title": "Normalized by true class",
            "format": ".3f",
            "vmax": 1.0,
            "colorbar_label": "Fraction",
        },
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5.5),
    )

    for ax, configuration in zip(
        axes,
        configurations,
        strict=True,
    ):
        matrix = configuration["matrix"]

        image = ax.imshow(
            matrix,
            cmap="Blues",
            vmin=0,
            vmax=configuration["vmax"],
        )

        add_matrix_text(
            ax,
            matrix,
            configuration["format"],
        )

        ax.set(
            xticks=np.arange(n_classes),
            yticks=np.arange(n_classes),
            xticklabels=class_names,
            yticklabels=class_names,
            xlabel="Predicted class",
            ylabel="True class",
            title=configuration["title"],
        )

        colorbar = fig.colorbar(
            image,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )
        colorbar.set_label(configuration["colorbar_label"])

    fig.suptitle(
        "Particle classification: confusion matrix",
        fontsize=15,
    )
    fig.tight_layout()

    save(
        fig,
        output_path(
            args.output_dir,
            args.output_file,
        ),
        args.dpi,
    )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
