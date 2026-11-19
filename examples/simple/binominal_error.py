import numpy as np

from analysis_tools.metrics import (
    binomial_error,
    binomial_errors,
    wilson_interval,
)

successes = np.array([0, 40, 80, 100])
trials = np.array([100, 50, 100, 100])

efficiency = successes / trials

standard_error = binomial_error(
    successes,
    trials,
)

wilson_lower, wilson_upper = wilson_interval(
    successes,
    trials,
)

lower_error, upper_error = binomial_errors(
    successes,
    trials,
    method="wilson",
)

print("Efficiency:")
print(efficiency)

print("\nStandard error:")
print(standard_error)

print("\nWilson interval:")
print("Lower:", wilson_lower)
print("Upper:", wilson_upper)

print("\nWilson error widths:")
print("Lower:", lower_error)
print("Upper:", upper_error)
