"""Partial distinguishability through explicit orthogonal internal modes."""
from dataclasses import dataclass
from itertools import product
from math import fsum, prod
import numpy as np
from .numerics import Precision, ResourceLimitError, density, finite_array, integer
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
    source_omitted_probability: float = 0.0
    input_model: str = "pure wavepackets"

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
                               self.gram_factorization_residual, self.source_omitted_probability,
                               self.input_model)

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


def _gram_coefficients(gram, photons, precision):
    g = finite_array(gram, 2)
    if g.shape != (photons,) * 2 or not np.allclose(g, g.conj().T, atol=precision.atol, rtol=0) or not np.allclose(np.diag(g), 1, atol=precision.atol, rtol=0):
        raise ValueError("Gram matrix must be Hermitian with a unit diagonal and one row per wavepacket")
    if np.array_equal(g, np.ones((photons, photons))):
        return np.ones((1, photons), complex), 0.
    if photons == 2 and np.array_equal(np.diag(g), [1, 1]) and abs(g[0, 1]) <= 1:
        # Stable two-packet construction: generic eigensolvers can have order-
        # epsilon absolute error in an eigenvalue much smaller than one.
        r = abs(g[0, 1])
        coefficients = np.array([[1, g[0, 1]], [0, np.sqrt((1-r)*(1+r))]], complex)
        if r == 1:
            coefficients = coefficients[:1]
        return coefficients, float(np.linalg.norm(coefficients.conj().T@coefficients-g))
    values, vectors = np.linalg.eigh(g)
    if values.min() < -precision.atol:
        raise ValueError("Gram matrix is not positive semidefinite")
    # atol validates the input; it is NOT a physical rank cutoff. Small positive
    # eigenvalues carry rare distinguishability events, including HOM coincidences.
    keep = values > 0
    coefficients = np.sqrt(values[keep])[:, None] * vectors[:, keep].conj().T
    residual = float(np.linalg.norm(coefficients.conj().T @ coefficients - g))
    return coefficients, residual


def _product_vector(basis, internal_modes, photons):
    """Normalized bosonic product of (spatial port, internal vector) pairs."""
    amplitudes = {(0,) * basis.modes: 1.0+0j}
    for spatial, coefficients in photons:
        updated = {}
        for occ, amp in amplitudes.items():
            for internal, coefficient in enumerate(coefficients):
                if coefficient == 0:
                    continue
                mode = spatial*internal_modes + internal
                dest = list(occ)
                dest[mode] += 1
                key = tuple(dest)
                updated[key] = updated.get(key, 0j) + amp*coefficient*np.sqrt(dest[mode])
        # Normalize each intermediate product to avoid factorial overflow at
        # large occupation. This only rescales the ket, not a source probability.
        norm = np.sqrt(fsum(abs(a)**2 for a in updated.values()))
        if norm == 0 or not np.isfinite(norm):
            raise ValueError("The specified bosonic input has invalid norm")
        amplitudes = {s: a/norm for s, a in updated.items()}
    norm = np.sqrt(fsum(abs(a)**2 for a in amplitudes.values()))
    if norm == 0:
        raise ValueError("The specified bosonic input has zero norm")
    vector = np.zeros(basis.dimension, complex)
    for occ, amplitude in amplitudes.items():
        vector[basis.index[occ]] = amplitude/norm
    return vector


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
    coefficients, residual = _gram_coefficients(gram, len(inputs), precision)
    rank = coefficients.shape[0]
    basis = FockBasis(spatial_modes*rank, len(inputs), precision=precision)
    state = FockState(basis, _product_vector(basis, rank,
                      [(port, coefficients[:, j]) for j, port in enumerate(inputs)]))
    return WavepacketState(state, spatial_modes, rank, residual)


def mixed_wavepacket_input(spatial_modes, input_modes, internal_states, *, precision=None):
    """Independent mixed internal states, one photon per DISTINCT input port.

    All matrices use the SAME orthonormal spectral/polarization basis. Embed
    their tensor product isometrically into Fock space, retaining every density
    entry. Pairwise purities alone do not determine three-photon interference.
    Correlated inputs and multiple photons at one port require a joint model.
    """
    precision = precision or Precision()
    spatial_modes = integer(spatial_modes, "spatial_modes", 1)
    inputs = tuple(integer(i, "input mode") for i in input_modes)
    if not inputs or len(set(inputs)) != len(inputs) or any(i >= spatial_modes for i in inputs):
        raise ValueError("Mixed inputs require distinct valid input ports")
    matrices = tuple(density(rho, atol=precision.atol) for rho in internal_states)
    if len(matrices) != len(inputs) or len({rho.shape for rho in matrices}) != 1:
        raise ValueError("One equally sized internal density matrix is required per photon")
    rank = len(matrices[0])
    basis = FockBasis(spatial_modes*rank, len(inputs), precision=precision)
    joint = np.array([[1.0+0j]])
    for rho in matrices:
        joint = np.kron(joint, rho)
    indices = []
    for internals in product(range(rank), repeat=len(inputs)):
        occ = [0]*basis.modes
        for port, internal in zip(inputs, internals):
            occ[port*rank+internal] = 1
        indices.append(basis.index[tuple(occ)])
    rho = np.zeros((basis.dimension, basis.dimension), complex)
    rho[np.ix_(indices, indices)] = joint
    return WavepacketState(FockState(basis, rho), spatial_modes, rank, 0.,
                           input_model="independent mixed internal density matrices")


def wavepacket_sources(spatial_modes, input_modes, distributions, gram, *,
                       multiphoton_model, max_photons=None, precision=None,
                       max_source_patterns=100000):
    """Independent number-diagonal sources with distinguishable wavepackets.

    Sources occupy distinct ports. gram describes the signal wavepacket of
    each source. Select multiphoton_model explicitly:
      same_wavepacket: all photons from a source share its signal wavepacket;
      orthogonal_noise: P(2) contains one signal and one noise photon. Each
        source's noise is orthogonal to all signals AND other sources' noise.
        This model requires per-source support n<=2.
    These are specified noise hypotheses, not bounds or an inference from g2.
    The total cutoff removes positive branches without renormalization.
    """
    from .sources import number_distribution
    precision = precision or Precision()
    spatial_modes = integer(spatial_modes, "spatial_modes", 1)
    inputs = tuple(integer(i, "input mode") for i in input_modes)
    if not inputs or len(set(inputs)) != len(inputs) or any(i >= spatial_modes for i in inputs):
        raise ValueError("Sources require distinct valid input ports")
    if multiphoton_model not in ("same_wavepacket", "orthogonal_noise"):
        raise ValueError("multiphoton_model must be same_wavepacket or orthogonal_noise")
    arrays = tuple(number_distribution(p, atol=precision.atol) for p in distributions)
    if len(arrays) != len(inputs):
        raise ValueError("One number distribution is required per source")
    support = [np.flatnonzero(p).tolist() for p in arrays]
    if multiphoton_model == "orthogonal_noise" and any(max(s) > 2 for s in support):
        raise ValueError("orthogonal_noise supports at most two photons per source")
    if prod(map(len, support)) > integer(max_source_patterns, "max_source_patterns", 1):
        raise ResourceLimitError("Source expansion exceeds max_source_patterns")
    cutoff = sum(max(s) for s in support) if max_photons is None else integer(max_photons, "max_photons")
    coefficients, residual = _gram_coefficients(gram, len(inputs), precision)
    signal_rank = coefficients.shape[0]
    noisy = [j for j, s in enumerate(support) if 2 in s] if multiphoton_model == "orthogonal_noise" else []
    rank = signal_rank+len(noisy)
    coefficients = np.pad(coefficients, ((0, len(noisy)), (0, 0)))
    noise_vectors = {j: np.eye(rank)[signal_rank+k] for k, j in enumerate(noisy)}
    basis = FockBasis(spatial_modes*rank, cutoff, precision=precision)
    rho = np.zeros((basis.dimension, basis.dimension), complex)
    omissions = []
    for counts in product(*support):
        weight = prod(p[n] for p, n in zip(arrays, counts))
        if sum(counts) > cutoff:
            omissions.append(weight)
            continue
        photons = []
        for j, (port, count) in enumerate(zip(inputs, counts)):
            photons.extend((port, noise_vectors[j] if k == 1 and j in noisy else coefficients[:, j])
                           for k in range(count))
        vector = _product_vector(basis, rank, photons)
        rho += weight*np.outer(vector, vector.conj())
    return WavepacketState(FockState(basis, rho, subnormalized=True), spatial_modes,
                           rank, residual, fsum(omissions), multiphoton_model)
