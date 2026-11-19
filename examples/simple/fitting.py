import numpy as np

from analysis_tools.fitting import fit_least_squares

RANDOM_SEED = 4242


def model(
    x: np.ndarray,
    slope: float,
    intercept: float,
) -> np.ndarray:
    """Return a linear model."""
    return slope * x + intercept


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    x = np.linspace(
        0.0,
        10.0,
        21,
    )
    yerr = np.full_like(
        x,
        0.2,
    )

    true_slope = 2.0
    true_intercept = 1.0

    y = model(
        x,
        slope=true_slope,
        intercept=true_intercept,
    )
    y += rng.normal(
        loc=0.0,
        scale=yerr,
    )

    result = fit_least_squares(
        model,
        x,
        y,
        yerr,
        initial={
            "slope": 1.0,
            "intercept": 0.0,
        },
        limits={
            "slope": (0.0, None),
        },
        run_minos=True,
    )

    print(f"Valid fit: {result.valid}")
    print(f"Slope: {result.value('slope'):.4f} +/- {result.error('slope'):.4f}")
    print(
        "Intercept: "
        f"{result.value('intercept'):.4f} "
        f"+/- {result.error('intercept'):.4f}"
    )
    print(f"Chi-square: {result.minimum_value:.3f}")
    print(f"Degrees of freedom: {result.degrees_of_freedom}")
    print(f"Chi-square / ndf: {result.reduced_minimum_value:.3f}")

    if "slope" in result.minos_errors:
        slope_minos = result.minos_errors["slope"]

        print(f"Slope MINOS error: {slope_minos.lower:+.4f}, {slope_minos.upper:+.4f}")


if __name__ == "__main__":
    main()
