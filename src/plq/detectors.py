"""Destructive photon counting with explicit efficiency, timing acceptance, and dark counts."""
from dataclasses import dataclass
from math import comb, erf, exp, factorial, sqrt
from itertools import product
import numpy as np
from scipy.stats import poisson
from .numerics import probability, integer
from .optics import FockBasis, FockState


@dataclass(frozen=True)
class Detector:
    efficiency: float = 1.0
    dark_rate_hz: float = 0.0
    gate_seconds: float = 1e-9
    threshold: bool = False
    saturation: int = 4
    arrival_offset_seconds: float = 0.0
    timing_sigma_seconds: float = 0.0

    def __post_init__(self):
        probability(self.efficiency, "efficiency")
        integer(self.saturation, "saturation", 1)
        if not isinstance(self.threshold, bool):
            raise ValueError("threshold must be Boolean")
        for value in (self.dark_rate_hz, self.gate_seconds, self.timing_sigma_seconds):
            if not np.isfinite(value) or value < 0:
                raise ValueError("Rates, gate duration, and timing sigma must be finite and nonnegative")
        if not np.isfinite(self.arrival_offset_seconds):
            raise ValueError("Arrival offset must be finite")
        if not np.isfinite(self.dark_rate_hz * self.gate_seconds):
            raise ValueError("Expected dark count is not finite")

    @property
    def effective_efficiency(self):
        half = self.gate_seconds / 2
        offset, sigma = self.arrival_offset_seconds, self.timing_sigma_seconds
        if self.gate_seconds == 0:
            acceptance = 0.0
        elif sigma == 0:
            acceptance = float(abs(offset) <= half)
        else:
            acceptance = (erf((half-offset)/(sqrt(2)*sigma)) - erf((-half-offset)/(sqrt(2)*sigma))) / 2
        return self.efficiency * max(0, min(1, acceptance))

    @property
    def outcomes(self):
        return tuple(range(2 if self.threshold else self.saturation+1))

    def response(self, photons):
        """P(reported count | photons). Last PNR bin is >= saturation; no discarded tail."""
        n = integer(photons, "photons")
        eta, mu = self.effective_efficiency, self.dark_rate_hz * self.gate_seconds
        if self.threshold:
            p0 = (1-eta)**n * exp(-mu)
            return np.array([p0, 1-p0])
        out = np.zeros(self.saturation + 1)
        for detected in range(n + 1):
            p = comb(n, detected) * eta**detected * (1-eta)**(n-detected)
            if detected >= self.saturation:
                out[-1] += p
                continue
            for count in range(detected, self.saturation):
                out[count] += p * poisson.pmf(count-detected, mu)
            out[-1] += p * poisson.sf(self.saturation-detected-1, mu)
        return out


def _setup(state, modes, detectors):
    modes = tuple(integer(m, "mode") for m in modes)
    if len(set(modes)) != len(modes) or any(m >= state.basis.modes for m in modes):
        raise ValueError("Invalid measurement modes")
    detectors = tuple(detectors) if detectors is not None else tuple(Detector(saturation=max(1, state.basis.max_photons+1)) for _ in modes)
    if len(detectors) != len(modes):
        raise ValueError("One detector is required per measured mode")
    responses = [{n: detector.response(n) for n in range(state.basis.max_photons+1)} for detector in detectors]
    return modes, detectors, responses


def detection_probabilities(state, modes=None, detectors=None, *, max_outcomes=100000):
    modes = tuple(range(state.basis.modes)) if modes is None else modes
    modes, detectors, responses = _setup(state, modes, detectors)
    if np.prod([len(d.outcomes) for d in detectors], dtype=object) > max_outcomes:
        from .numerics import ResourceLimitError
        raise ResourceLimitError("Joint detector outcome count exceeds max_outcomes")
    result = {}
    for counts in product(*(d.outcomes for d in detectors)):
        result[counts] = float(sum(state.rho[i, i].real * np.prod([responses[j][occ[m]][counts[j]]
                                    for j, m in enumerate(modes)]) for i, occ in enumerate(state.basis.states)))
    return result


@dataclass
class HeraldResult:
    probability: float
    remaining_state: FockState | None
    measured_modes: tuple
    remaining_modes: tuple

    def conditional_state(self):
        if self.probability <= 0:
            raise ValueError("The requested herald has zero probability")
        if self.remaining_state is None:
            raise ValueError("All modes were destructively measured")
        return self.remaining_state.conditional()


def herald(state, modes, counts, detectors=None):
    """Absorb measured modes, trace their environments, and retain the absolute branch weight.

    Number-diagonal POVM; not a square-root nondestructive measurement instrument.
    """
    modes, detectors, responses = _setup(state, modes, detectors)
    counts = tuple(integer(c, "count") for c in counts)
    if len(counts) != len(modes) or any(c not in d.outcomes for c, d in zip(counts, detectors)):
        raise ValueError("Counts do not match detector outcomes")
    remaining = tuple(m for m in range(state.basis.modes) if m not in modes)
    groups = {}
    for i, occ in enumerate(state.basis.states):
        measured = tuple(occ[m] for m in modes)
        groups.setdefault(measured, []).append(i)
    if not remaining:
        p = sum(float(state.rho[i, i].real) * np.prod([responses[j][occ[m]][counts[j]] for j, m in enumerate(modes)])
                for i, occ in enumerate(state.basis.states))
        return HeraldResult(float(p), None, modes, remaining)
    basis = FockBasis(len(remaining), state.basis.max_photons, precision=state.basis.precision)
    rho = np.zeros((basis.dimension, basis.dimension), complex)
    for measured, indices in groups.items():
        weight = np.prod([responses[j][n][counts[j]] for j, n in enumerate(measured)])
        dest = [basis.index[tuple(state.basis.states[i][m] for m in remaining)] for i in indices]
        rho[np.ix_(dest, dest)] += weight * state.rho[np.ix_(indices, indices)]
    output = FockState(basis, rho, subnormalized=True)
    return HeraldResult(output.trace, output, modes, remaining)
