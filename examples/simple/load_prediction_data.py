from pathlib import Path

from analysis_tools.io import (
    load_prediction_data,
)

PREDICTIONS_PATH = Path("path/to/outputs/preds.npy")

TARGETS_PATH = Path("path/to/outputs/targets.npy")

# Set to None when no event-index file is needed.
INDICES_PATH: Path | None = Path("path/to/outputs/indices.npy")


data = load_prediction_data(
    PREDICTIONS_PATH,
    TARGETS_PATH,
    INDICES_PATH,
)

print(f"Events: {data.event_count:,}")
print(f"Prediction shape: {data.predictions.shape}")
print(f"Target shape: {data.targets.shape}")

if data.indices is not None:
    print(f"Index shape: {data.indices.shape}")
    print(
        "First indices:",
        data.indices[:5],
    )
