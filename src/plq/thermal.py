"""Thermal attenuation by a beam-splitter dilation with an explicit bath tail.

The system basis grows by the bath cutoff. No system occupation is discarded
between operations, so photons can return from high-number sectors later.
"""
from dataclasses import dataclass
from math import ceil, exp, log, log1p
import numpy as np
from scipy.linalg import expm
from .numerics import integer, probability, ResourceLimitError


@dataclass(frozen=True)
class ThermalBath:
    mean_photons: float
    cutoff: int
    omitted_probability: float

    @classmethod
    def create(cls, mean_photons, bath_cutoff=None, tail_tolerance=1e-12):
        if (not np.isscalar(mean_photons) or not np.isreal(mean_photons)
                or not np.isfinite(mean_photons) or mean_photons < 0):
            raise ValueError("mean_photons must be finite, real and nonnegative")
        mean = float(mean_photons)
        tol = probability(tail_tolerance, "tail_tolerance")
        if not 0 < tol < 1:
            raise ValueError("tail_tolerance must be strictly between zero and one")
        if bath_cutoff is not None:
            bath_cutoff = integer(bath_cutoff, "bath_cutoff")
        if mean == 0:
            return cls(0., 0, 0.)
        log_q = log(mean)-log1p(mean) if mean < 1 else -log1p(1/mean)
        if bath_cutoff is None:
            needed = log(tol)/log_q
            if not np.isfinite(needed):
                raise ResourceLimitError("Thermal bath cutoff is beyond finite numerical resources")
            bath_cutoff = max(0, ceil(needed)-1)
            if exp((bath_cutoff+1)*log_q) > tol:
                bath_cutoff += 1
        return cls(mean, bath_cutoff, exp((bath_cutoff+1)*log_q))

    def weights(self):
        if self.mean_photons == 0:
            return np.array([1.])
        log_q = (log(self.mean_photons)-log1p(self.mean_photons)
                 if self.mean_photons < 1 else -log1p(1/self.mean_photons))
        return np.exp(np.arange(self.cutoff+1)*log_q-log1p(self.mean_photons))


def _paths(basis, mode, transmission, bath):
    """Yield sparse Kraus index triples, labelled by bath input/output numbers."""
    from .optics import FockBasis
    nmax, cutoff = basis.max_photons, bath.cutoff
    output = FockBasis(basis.modes, nmax+cutoff, precision=basis.precision)
    # Bound the coefficient tensor before allocating it. Only one number-
    # conserving beam-splitter block is exponentiated at a time.
    basis.precision.guard_kraus(nmax+1, cutoff+1, nmax+cutoff+1)
    amplitudes = np.zeros((nmax+1, cutoff+1, nmax+cutoff+1))
    theta = np.arccos(np.sqrt(transmission))
    for total in range(nmax+cutoff+1):
        coupling = np.sqrt(np.arange(1, total+1)*np.arange(total, 0, -1))
        generator = np.diag(coupling, 1)-np.diag(coupling, -1)
        rotation = expm(theta*generator)
        for n in range(max(0, total-cutoff), min(nmax, total)+1):
            amplitudes[n, total-n, :total+1] = rotation[::-1, n]
    weights = bath.weights()
    for incoming, weight in enumerate(weights):
        for outgoing in range(nmax+incoming+1):
            source, dest, coefficients = [], [], []
            for col, occ in enumerate(basis.states):
                survivors = occ[mode]+incoming-outgoing
                if survivors < 0:
                    continue
                coefficient = np.sqrt(weight)*amplitudes[occ[mode], incoming, outgoing]
                if coefficient == 0:
                    continue
                target = list(occ)
                target[mode] = survivors
                source.append(col)
                dest.append(output.index[tuple(target)])
                coefficients.append(coefficient)
            if source:
                yield output, source, dest, np.asarray(coefficients)


def apply_thermal_loss(rho, basis, mode, transmission, bath):
    """Return (rho, enlarged basis); retained trace equals input trace*(1-tail)."""
    from .optics import FockBasis, apply_loss
    if transmission == 1:
        return rho.copy(), basis
    if bath.mean_photons == 0:
        return apply_loss(rho, basis, mode, transmission), basis
    output = FockBasis(basis.modes, basis.max_photons+bath.cutoff, precision=basis.precision)
    result = np.zeros((output.dimension, output.dimension), complex)
    for _, source, dest, c in _paths(basis, mode, transmission, bath):
        result[np.ix_(dest, dest)] += rho[np.ix_(source, source)]*np.outer(c, c)
    return result, output


def thermal_loss_operators(basis, mode, transmission, bath):
    """Explicit rectangular Kraus operators for small thermal instruments."""
    from .optics import FockBasis, loss_operators
    if transmission == 1:
        return [np.eye(basis.dimension)], basis
    if bath.mean_photons == 0:
        return loss_operators(basis, mode, transmission), basis
    output = FockBasis(basis.modes, basis.max_photons+bath.cutoff, precision=basis.precision)
    count = sum(basis.max_photons+k+1 for k in range(bath.cutoff+1))
    basis.precision.guard_kraus(output.dimension, basis.dimension, count)
    operators = []
    for _, source, dest, c in _paths(basis, mode, transmission, bath):
        k = np.zeros((output.dimension, basis.dimension), complex)
        k[dest, source] = c
        operators.append(k)
    return operators, output
