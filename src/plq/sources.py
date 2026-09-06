"""Number statistics with explicit moment and photon-cutoff conventions."""
from math import fsum
import numpy as np
from .numerics import probability


def number_distribution(values, *, atol=1e-10):
    """Validate an explicit normalized P(n); never normalize a missing tail."""
    p = np.array([probability(value, "number probability") for value in values])
    if not len(p) or abs(fsum(p)-1) > atol:
        raise ValueError("Number probabilities must be nonempty and sum to one")
    return p


def number_moments(values):
    """Return mean photon number and normally ordered g2 (undefined for vacuum)."""
    p = number_distribution(values)
    mean = fsum(n*weight for n, weight in enumerate(p))
    factorial_second = fsum(n*(n-1)*weight for n, weight in enumerate(p))
    return {"mean_photons": mean, "factorial_second_moment": factorial_second,
            "g2_zero": None if mean == 0 else (factorial_second/mean)/mean}


def number_distribution_from_moments(mean_photons, g2_zero):
    """Infer P(0), P(1), P(2) ONLY under the explicit assumption P(n>=3)=0.

    mean_photons is <n>, not extraction efficiency, P(1), or P(n>0).
    g2_zero = <n(n-1)>/<n>**2 is not a two-photon probability or an
    uncorrected finite-efficiency HBT click ratio. These two moments do not
    identify the number distribution without the stated support assumption.
    """
    for name, value in (("mean_photons", mean_photons), ("g2_zero", g2_zero)):
        if (not np.isscalar(value) or not np.isrealobj(value)
                or not np.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite, real and nonnegative")
    mean, g2 = float(mean_photons), float(g2_zero)
    if not 0 < mean <= 2:
        raise ValueError("Moment inference requires 0 < mean_photons <= 2; use [1] for vacuum")
    # Check the physical support before multiplication to avoid overflow.
    if g2 > 1/mean:
        raise ValueError("Moments require negative P(1) under the n<=2 assumption")
    p2 = .5*(g2*mean)*mean
    if g2 > 0 and p2 == 0:
        raise ValueError("P(2) underflows float64; rescale the scenario or use an arbitrary-precision model")
    p1 = mean-2*p2
    p0 = 1-mean+p2
    if p0 < 0 or p1 < 0:
        raise ValueError("Moments are incompatible with a nonnegative n<=2 distribution")
    return np.array([p0, p1, p2])
