from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis_tools.fitting import fit_least_squares
from analysis_tools.plotting import save_figure

# Settings
RANDOM_SEED = 4242
NUMBER_OF_EVENTS = 10_000

TRUE_MEAN = 3.0
TRUE_SIGMA = 35.0

FIT_RANGE = (-120.0, 120.0)
NUMBER_OF_BINS = 60

OUTPUT_PATH = Path(
    "outputs/fitting/gaussian_fitting.png"
)

SHOW_FIGURE = True
DPI = 200


def gaussian(
    x: np.ndarray,
    amplitude: float,
    mean: float,
    sigma: float,
) -> np.ndarray:
    """Return a Gaussian model evaluated at x."""
    return amplitude * np.exp(
        -0.5 * ((x - mean) / sigma) ** 2
    )


def main() -> None:
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    # Replace this synthetic array with a residual array when
    # applying the example to real reconstruction results.
    residuals = rng.normal(
        loc=TRUE_MEAN,
        scale=TRUE_SIGMA,
        size=NUMBER_OF_EVENTS,
    )

    selected = residuals[
        (residuals >= FIT_RANGE[0])
        & (residuals <= FIT_RANGE[1])
    ]

    counts, edges = np.histogram(
        selected,
        bins=NUMBER_OF_BINS,
        range=FIT_RANGE,
    )

    centers = 0.5 * (
        edges[:-1] + edges[1:]
    )

    # Poisson uncertainty for each histogram bin.
    #
    # Empty bins cannot be assigned zero uncertainty in a
    # least-squares fit, so they are excluded here.
    valid = counts > 0

    fit_x = centers[valid]
    fit_y = counts[valid].astype(float)
    fit_yerr = np.sqrt(fit_y)

    result = fit_least_squares(
        gaussian,
        fit_x,
        fit_y,
        fit_yerr,
        initial={
            "amplitude": float(
                np.max(fit_y)
            ),
            "mean": float(
                np.mean(selected)
            ),
            "sigma": float(
                np.std(selected)
            ),
        },
        limits={
            "amplitude": (0.0, None),
            "sigma": (1e-12, None),
        },
        run_minos=True,
        minos_parameters=[
            "mean",
            "sigma",
        ],
    )

    figure, axis = plt.subplots(
        figsize=(7.0, 5.0)
    )

    axis.errorbar(
        centers,
        counts,
        yerr=np.sqrt(counts),
        fmt="o",
        color="black",
        markersize=4.0,
        capsize=2.0,
        label="Synthetic residuals",
    )

    fit_curve_x = np.linspace(
        FIT_RANGE[0],
        FIT_RANGE[1],
        500,
    )
    fit_curve_y = gaussian(
        fit_curve_x,
        amplitude=result.value("amplitude"),
        mean=result.value("mean"),
        sigma=result.value("sigma"),
    )

    chi_square = result.minimum_value
    ndf = result.degrees_of_freedom
    reduced_chi_square = result.reduced_minimum_value

    chi_square_text = (
        rf"$\chi^2 / \mathrm{{ndf}}"
        rf" = {chi_square:.1f} / {ndf}"
        rf" = {reduced_chi_square:.2f}$"
    )

    axis.plot(
        fit_curve_x,
        fit_curve_y,
        color="tab:red",
        linewidth=2.0,
        label=(
            "Gaussian fit\n"
            f"mean = "
            f"{result.value('mean'):.2f} "
            f"+/- {result.error('mean'):.2f}\n"
            f"sigma = "
            f"{result.value('sigma'):.2f} "
            f"+/- {result.error('sigma'):.2f}\n"
            f"{chi_square_text}"
        ),
    )


    axis.set_xlabel("Residual")
    axis.set_ylabel("Events per bin")
    axis.set_title(
        "Gaussian fit with iminuit"
    )
    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()

    print(
        f"Events in fit range: "
        f"{len(selected):,}"
    )
    print(f"Valid fit: {result.valid}")
    print(
        "Mean: "
        f"{result.value('mean'):.6g} "
        f"+/- {result.error('mean'):.6g}"
    )
    print(
        "Sigma: "
        f"{result.value('sigma'):.6g} "
        f"+/- {result.error('sigma'):.6g}"
    )
    print(
        "Chi-square / ndf: "
        f"{chi_square:.1f} / {ndf} "
        f"= {reduced_chi_square:.2f}"
    )

    for parameter in (
        "mean",
        "sigma",
    ):
        if parameter not in result.minos_errors:
            continue

        error = result.minos_errors[
            parameter
        ]

        print(
            f"{parameter} MINOS: "
            f"{error.lower:+.6g}, "
            f"{error.upper:+.6g}"
        )

    save_figure(
        figure,
        OUTPUT_PATH,
        dpi=DPI,
        close=not SHOW_FIGURE,
    )

    print(f"Saved: {OUTPUT_PATH}")

    if SHOW_FIGURE:
        plt.show()


if __name__ == "__main__":
    main()