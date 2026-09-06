"""Sparse pure-state optics and explicit quantum-trajectory sampling.

No Fock density matrix or lifted D x D unitary is allocated. Complexity still
follows occupied Fock support; this is not a polynomial boson-sampling solver.
"""
from dataclasses import dataclass, asdict
from math import comb
import numpy as np
from .numerics import integer, ResourceLimitError
from .simulation import wilson_interval


@dataclass(frozen=True)
class SparseBudget:
    max_photons: int = 256
    max_terms: int = 200_000
    max_transitions: int = 10_000_000

    def __post_init__(self):
        integer(self.max_photons, "max_photons")
        integer(self.max_terms, "max_terms", 1)
        integer(self.max_transitions, "max_transitions", 1)


class _Work:
    def __init__(self, budget):
        self.budget, self.transitions, self.peak = budget, 0, 0

    def check(self, size, transitions=0):
        self.transitions += transitions
        self.peak = max(self.peak, size)
        if size > self.budget.max_terms or self.transitions > self.budget.max_transitions:
            raise ResourceLimitError("Sparse support/work exceeds max_terms or max_transitions; no terms were truncated")


class SparseKet:
    """Normalized amplitudes keyed by occupations, including number superpositions."""
    def __init__(self, modes, amplitudes, *, budget=None):
        self.modes = integer(modes, "modes", 1)
        self.budget = budget or SparseBudget()
        if len(amplitudes) > self.budget.max_terms:
            raise ResourceLimitError("Input exceeds max_terms")
        self.amplitudes = {}
        for occupation, amplitude in amplitudes.items():
            occ = tuple(integer(n, "occupation") for n in occupation)
            amp = complex(amplitude)
            if len(occ) != self.modes or not np.isfinite(amp):
                raise ValueError("Invalid occupation or nonfinite amplitude")
            if sum(occ) > self.budget.max_photons:
                raise ResourceLimitError("Input exceeds sparse max_photons")
            if amp != 0:
                self.amplitudes[occ] = amp
        if not np.isclose(self.norm, 1, atol=1e-10, rtol=0):
            raise ValueError("SparseKet must have norm squared one")

    @classmethod
    def occupation(cls, occupation, *, budget=None):
        occupation = tuple(occupation)
        return cls(len(occupation), {occupation: 1}, budget=budget)

    @property
    def norm(self):
        return float(sum(abs(a)**2 for a in self.amplitudes.values()))

    def probabilities(self):
        return {s: float(abs(a)**2) for s, a in self.amplitudes.items()}

    def to_fock(self, *, precision=None):
        """Explicit budgeted conversion, intended for small reference checks."""
        from .optics import FockBasis, FockState
        basis = FockBasis(self.modes, max(map(sum, self.amplitudes)), precision=precision)
        return FockState.amplitudes(basis, self.amplitudes)


def _unitary(amplitudes, matrix, work):
    # Circuit stores a mode-space matrix; infer exact identity spectators.
    changed = matrix != np.eye(len(matrix))
    targets = np.flatnonzero(np.any(changed, axis=0) | np.any(changed, axis=1))
    if not len(targets):
        return amplitudes.copy()
    u = matrix[np.ix_(targets, targets)]
    cache, output = {}, {}
    for source, amplitude in amplitudes.items():
        local = tuple(source[m] for m in targets)
        if local not in cache:
            expansion = {(0,)*len(targets): 1+0j}
            for mode, count in enumerate(local):
                for k in range(1, count+1):
                    updated = {}
                    for occ, amp in expansion.items():
                        for dest in np.flatnonzero(u[:, mode]):
                            nxt = list(occ)
                            nxt[dest] += 1
                            key = tuple(nxt)
                            updated[key] = updated.get(key, 0j) + amp*u[dest, mode]*np.sqrt(nxt[dest]/k)
                            work.check(len(updated), 1)
                    expansion = updated
            cache[local] = expansion
            work.check(sum(len(v) for v in cache.values()))
        for local_out, amp in cache[local].items():
            dest = list(source)
            for mode, count in zip(targets, local_out):
                dest[mode] = count
            key = tuple(dest)
            output[key] = output.get(key, 0j) + amplitude*amp
            work.check(len(output), 1)
    return {s: a for s, a in output.items() if a != 0}


def _loss(amplitudes, mode, eta, rng):
    # Sample an environment photon count with its Born probability, then apply
    # that loss Kraus operator. No no-loss conditioning of the final ensemble.
    largest = max(s[mode] for s in amplitudes)
    weights = np.zeros(largest+1)
    for occ, amp in amplitudes.items():
        n = occ[mode]
        for lost in range(n+1):
            weights[lost] += abs(amp)**2 * comb(n, lost)*(1-eta)**lost*eta**(n-lost)
    lost = int(rng.choice(len(weights), p=weights/weights.sum()))
    output = {}
    for occ, amp in amplitudes.items():
        n = occ[mode]
        if n >= lost:
            coefficient = np.sqrt(comb(n, lost)*(1-eta)**lost*eta**(n-lost)/weights[lost])
            if coefficient:
                dest = list(occ)
                dest[mode] -= lost
                output[tuple(dest)] = amp*coefficient
    return output


def _prepare(circuit, state, budget):
    state = state if isinstance(state, SparseKet) else SparseKet.occupation(state, budget=budget)
    if circuit.modes != state.modes:
        raise ValueError("Input and circuit modes differ")
    if any(kind not in ("unitary", "loss", "phase_noise") for kind, _ in circuit.steps):
        raise ValueError("Sparse backend supports passive optics, vacuum loss and Gaussian phase noise; thermal/mixed inputs need density backend")
    budget = budget or state.budget
    if len(state.amplitudes) > budget.max_terms or max(map(sum, state.amplitudes)) > budget.max_photons:
        raise ResourceLimitError("Input exceeds the requested sparse budget")
    return state, budget


def _propagate(circuit, state, budget, rng=None):
    work, amplitudes = _Work(budget), state.amplitudes.copy()
    work.check(len(amplitudes))
    for kind, data in circuit.steps:
        if kind == "unitary":
            amplitudes = _unitary(amplitudes, data, work)
        elif rng is None:
            raise ValueError("Noisy circuits require sample_trajectories; run_sparse is a pure lossless calculation")
        elif kind == "loss":
            amplitudes = _loss(amplitudes, *data, rng)
        else:
            covariance, mean = data
            phases = rng.multivariate_normal(mean.real, covariance.real, check_valid="raise")
            amplitudes = {s: a*np.exp(1j*np.dot(s, phases)) for s, a in amplitudes.items()}
    return SparseKet(state.modes, amplitudes, budget=budget), work


def run_sparse(circuit, state, *, budget=None):
    """Exact lossless complex128 propagation; no amplitude threshold or cutoff."""
    state, budget = _prepare(circuit, state, budget)
    return _propagate(circuit, state, budget)[0]


@dataclass
class TrajectoryResult:
    shots: int
    seed: int
    counts: dict
    peak_terms: int
    max_transitions_per_shot: int
    budget: dict
    detector_counts: dict | None = None
    model: str = "Monte Carlo pure-state loss/phase trajectories; number measurement; not exact probabilities"

    def interval(self, occupation):
        return wilson_interval(self.counts.get(tuple(occupation), 0), self.shots)

    def to_dict(self):
        result = asdict(self)
        result["counts"] = [{"occupation": list(k), "count": v, "probability_estimate": v/self.shots,
                             "wilson_95": self.interval(k)} for k, v in sorted(self.counts.items())]
        if self.detector_counts is not None:
            result["detector_counts"] = [{"counts": list(k), "count": v} for k, v in sorted(self.detector_counts.items())]
        return result


def sample_trajectories(circuit, state, *, shots=1000, seed=0, budget=None, detectors=None):
    """One trajectory at a time; samples every photon-loss sector, including vacuum.

    Detector responses are sampled independently per shot. Dead time and
    inter-pulse correlations are not modeled. Returned intervals are marginal,
    not simultaneous; use interval(occupation) also for unobserved outcomes.
    """
    state, budget = _prepare(circuit, state, budget)
    shots, seed = integer(shots, "shots", 1), integer(seed, "seed")
    if detectors is not None and len(detectors) != circuit.modes:
        raise ValueError("One detector per output mode is required")
    rng = np.random.default_rng(seed)
    counts, detected, peak, transitions = {}, {}, 0, 0
    lossless = all(kind == "unitary" for kind, _ in circuit.steps)
    cached = _propagate(circuit, state, budget) if lossless else None
    for _ in range(shots):
        output, work = cached or _propagate(circuit, state, budget, rng)
        peak, transitions = max(peak, work.peak), max(transitions, work.transitions)
        occupations = tuple(output.amplitudes)
        p = np.array(list(output.probabilities().values()))
        measured = occupations[int(rng.choice(len(p), p=p/p.sum()))]
        counts[measured] = counts.get(measured, 0)+1
        if detectors is not None:
            pattern = tuple(int(rng.choice(len(d.outcomes), p=d.response(n))) for d, n in zip(detectors, measured))
            detected[pattern] = detected.get(pattern, 0)+1
    return TrajectoryResult(shots, seed, counts, peak, transitions, asdict(budget),
                            detected if detectors is not None else None)
