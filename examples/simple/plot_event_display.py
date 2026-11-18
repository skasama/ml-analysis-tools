"""Create simple 2D and 3D event displays using synthetic PMT data.

Run:

    uv run python examples/simple/plot_event_display.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis_tools.plotting import (
    add_cylindrical_detector_3d,
    plot_event_3d,
    plot_unrolled_cylindrical_event,
    save_figure,
)

# ============================================================
# Output
# ============================================================

OUTPUT_DIRECTORY = Path("outputs/event_display")

DISPLAY_2D_FILENAME = "event_display_2d.png"
DISPLAY_3D_FILENAME = "event_display_3d.png"

DPI = 200
SHOW_FIGURES = True


# ============================================================
# Synthetic detector settings
# ============================================================

RANDOM_SEED = 4242

DETECTOR_RADIUS = 10.0
DETECTOR_HALF_HEIGHT = 12.0

NUMBER_OF_BARREL_PMTS = 800
NUMBER_OF_ENDCAP_PMTS = 200
NUMBER_OF_HITS = 80


def generate_cylindrical_pmt_positions(
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate synthetic PMT positions on a cylindrical detector."""
    barrel_phi = rng.uniform(
        -np.pi,
        np.pi,
        NUMBER_OF_BARREL_PMTS,
    )
    barrel_z = rng.uniform(
        -DETECTOR_HALF_HEIGHT,
        DETECTOR_HALF_HEIGHT,
        NUMBER_OF_BARREL_PMTS,
    )

    barrel_positions = np.column_stack(
        (
            DETECTOR_RADIUS * np.cos(barrel_phi),
            DETECTOR_RADIUS * np.sin(barrel_phi),
            barrel_z,
        )
    )

    endcap_phi = rng.uniform(
        -np.pi,
        np.pi,
        NUMBER_OF_ENDCAP_PMTS,
    )
    endcap_radius = DETECTOR_RADIUS * np.sqrt(
        rng.uniform(
            0.0,
            1.0,
            NUMBER_OF_ENDCAP_PMTS,
        )
    )
    endcap_z = np.where(
        np.arange(NUMBER_OF_ENDCAP_PMTS) % 2 == 0,
        DETECTOR_HALF_HEIGHT,
        -DETECTOR_HALF_HEIGHT,
    )

    endcap_positions = np.column_stack(
        (
            endcap_radius * np.cos(endcap_phi),
            endcap_radius * np.sin(endcap_phi),
            endcap_z,
        )
    )

    return np.concatenate(
        (
            barrel_positions,
            endcap_positions,
        ),
        axis=0,
    )


def generate_synthetic_event(
    pmt_positions: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Select hit PMTs and generate synthetic charge values."""
    hit_indices = rng.choice(
        len(pmt_positions),
        size=NUMBER_OF_HITS,
        replace=False,
    )

    hit_positions = pmt_positions[hit_indices]

    direction = np.array(
        [0.7, 0.2, 0.68],
        dtype=float,
    )
    direction /= np.linalg.norm(direction)

    projected_distance = hit_positions @ direction
    projected_distance -= projected_distance.min()

    expected_charge = 1.0 + 20.0 * np.exp(
        -0.5 * ((projected_distance - 8.0) / 3.0) ** 2
    )

    charges = rng.gamma(
        shape=2.0,
        scale=expected_charge / 2.0,
    )

    return hit_positions, charges


def cylindrical_projection(
    positions: np.ndarray,
) -> np.ndarray:
    """Project 3D positions onto azimuth-z coordinates."""
    azimuth = np.arctan2(
        positions[:, 1],
        positions[:, 0],
    )

    return np.column_stack(
        (
            azimuth,
            positions[:, 2],
        )
    )


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    pmt_positions = generate_cylindrical_pmt_positions(rng)
    hit_positions, charges = generate_synthetic_event(
        pmt_positions,
        rng,
    )

    figure_2d, _ = plot_unrolled_cylindrical_event(
        charges,
        hit_positions,
        all_pmt_coordinates=pmt_positions,
        radius=DETECTOR_RADIUS,
        half_height=DETECTOR_HALF_HEIGHT,
        axis=2,
        title="Synthetic cylindrical event display",
        color_label="Charge [a.u.]",
        cmap="plasma",
        marker_size=24.0,
        show_grid=True,
    )

    figure_3d, axis_3d = plot_event_3d(
        charges,
        hit_positions,
        all_pmt_coordinates=pmt_positions,
        title="Synthetic 3D event display",
        color_label="Charge [a.u.]",
        show_axes=True,
        show_grid=True,
        show_background_panes=True,
        equal_aspect=True,
    )

    add_cylindrical_detector_3d(
        axis_3d,
        radius=DETECTOR_RADIUS,
        half_height=DETECTOR_HALF_HEIGHT,
        axis=2,
    )

    output_2d = OUTPUT_DIRECTORY / DISPLAY_2D_FILENAME
    output_3d = OUTPUT_DIRECTORY / DISPLAY_3D_FILENAME

    save_figure(
        figure_2d,
        output_2d,
        dpi=DPI,
        close=not SHOW_FIGURES,
    )
    save_figure(
        figure_3d,
        output_3d,
        dpi=DPI,
        close=not SHOW_FIGURES,
    )

    print(f"PMTs: {len(pmt_positions):,}")
    print(f"Hits: {len(hit_positions):,}")
    print(f"Saved: {output_2d}")
    print(f"Saved: {output_3d}")

    if SHOW_FIGURES:
        plt.show()


if __name__ == "__main__":
    main()
