"""Partial distinguishability through explicit orthogonal internal modes."""
from dataclasses import dataclass
import numpy as np
from .numerics import Precision, finite_array, integer
from .optics import FockBasis, FockState, Circuit


def gaussian_gram(arrival_seconds, angular_frequencies, sigma_seconds):
    """Overlap of normalized equal-width pure Gaussian temporal amplitudes.

    psi_j(t) ∝ exp[-(t-t_j)^2/(4 sigma^2)] exp[-i omega_j t].
    sigma is the intensity standard deviation; omega uses rad/s. The common
    carrier frequency can be subtracted to improve numerical conditioning.
    """
    t = finite_array(arrival_seconds)
    w = finite_array(angular_frequencies)
    if np.any(t.imag != 0) or np.any(w.imag != 0):
        raise ValueError("Arrival times and angular frequencies must be real")
    t, w = t.real, w.real
    if t.ndim != 1 or len(t) == 0 or t.shape != w.shape or not np.all(np.isfinite(t)) or not np.all(np.isfinite(w)):
        raise ValueError("Arrival times and angular frequencies must be finite equally sized vectors")
    if not np.isfinite(sigma_seconds) or sigma_seconds <= 0:
        raise ValueError("sigma_seconds must be finite and positive")
    dt, dw = t[:, None] - t[None, :], w[:, None] - w[None, :]
    return np.exp(-dt**2/(8*sigma_seconds**2) - sigma_seconds**2*dw**2/2
                  + 0.5j * dw * (t[:, None] + t[None, :]))


@dataclass
class WavepacketState:
    state: FockState
    spatial_modes: int
    internal_modes: int
    gram_factorization_residual: float

    def through(self, circuit):
        if circuit.modes != self.spatial_modes:
            raise ValueError("Spatial mode counts differ")
        expanded = Circuit(self.spatial_modes * self.internal_modes)
        for kind, data in circuit.steps:
            if kind == "unitary":
                expanded.unitary(np.kron(data, np.eye(self.internal_modes)))
            elif kind == "loss":
                mode, eta = data
                for j in range(self.internal_modes):
                    expanded.loss(mode*self.internal_modes+j, eta)
            elif kind == "phase_noise":
                covariance, mean = data
                expanded.phase_noise(np.kron(covariance, np.ones((self.internal_modes,)*2)),
                                     np.repeat(mean, self.internal_modes))
            else:
                raise ValueError(f"Unsupported operation {kind}")
        return WavepacketState(expanded.run(self.state).state, self.spatial_modes, self.internal_modes,
                               self.gram_factorization_residual)

    def spatial_probabilities(self):
        result = {}
        for occupation, p in self.state.probabilities().items():
            spatial = tuple(sum(occupation[m*self.internal_modes:(m+1)*self.internal_modes]) for m in range(self.spatial_modes))
            result[spatial] = result.get(spatial, 0.0) + p
        return result

    def detection_probabilities(self, detectors, *, max_outcomes=100000):
        from itertools import product
        if len(detectors) != self.spatial_modes:
            raise ValueError("One detector per spatial mode is required")
        if np.prod([len(d.outcomes) for d in detectors], dtype=object) > max_outcomes:
            from .numerics import ResourceLimitError
            raise ResourceLimitError("Detector outcome count exceeds max_outcomes")
        spatial = self.spatial_probabilities()
        responses = [[d.response(n) for n in range(self.state.basis.max_photons+1)] for d in detectors]
        return {out: float(sum(p * np.prod([responses[j][n][out[j]] for j, n in enumerate(occ)])
                               for occ, p in spatial.items())) for out in product(*(d.outcomes for d in detectors))}


def wavepacket_input(spatial_modes, input_modes, gram, *, precision=None):
    """One pure internal wavepacket per photon, arbitrary valid COMPLEX Gram matrix.

    Internal modes remain explicit throughout the calculation. Returning only a
    spatial count distribution does not create a fictitious coherent spatial ket.
    """
    precision = precision or Precision()
    spatial_modes = integer(spatial_modes, "spatial_modes", 1)
    inputs = tuple(integer(i, "input mode") for i in input_modes)
    if not inputs or any(i >= spatial_modes for i in inputs):
        raise ValueError("At least one valid input photon is required")
    g = finite_array(gram, 2)
    if g.shape != (len(inputs),) * 2 or not np.allclose(g, g.conj().T, atol=precision.atol, rtol=0) or not np.allclose(np.diag(g), 1, atol=precision.atol, rtol=0):
        raise ValueError("Gram matrix must be Hermitian with a unit diagonal and one row per photon")
    values, vectors = np.linalg.eigh(g)
    if values.min() < -precision.atol:
        raise ValueError("Gram matrix is not positive semidefinite")
    keep = values > precision.atol
    coefficients = np.sqrt(values[keep])[:, None] * vectors[:, keep].conj().T
    residual = float(np.linalg.norm(coefficients.conj().T @ coefficients - g))
    rank = coefficients.shape[0]
    basis = FockBasis(spatial_modes*rank, len(inputs), precision=precision)
    amplitudes = {(0,) * basis.modes: 1.0+0j}
    for photon, spatial in enumerate(inputs):
        updated = {}
        for occ, amp in amplitudes.items():
            for internal in range(rank):
                mode = spatial*rank + internal
                dest = list(occ)
                dest[mode] += 1
                key = tuple(dest)
                updated[key] = updated.get(key, 0j) + amp * coefficients[internal, photon] * np.sqrt(dest[mode])
        amplitudes = updated
    # The symmetrized product must be normalized, especially for repeated input ports.
    norm = np.sqrt(sum(abs(a)**2 for a in amplitudes.values()))
    if norm == 0:
        raise ValueError("The specified bosonic input has zero norm")
    state = FockState.amplitudes(basis, {s: a/norm for s, a in amplitudes.items()})
    return WavepacketState(state, spatial_modes, rank, residual)
