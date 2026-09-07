"""Passive linear optics, pure loss, and Gaussian phase diffusion in Fock space."""
from dataclasses import dataclass
from functools import cached_property
from math import comb, factorial
from itertools import product
import numpy as np
from .numerics import Precision, ResourceLimitError, density, diagnostics, finite_array, integer, probability, unitary
from .channels import Branch, KrausChannel


def compositions(total, modes):
    if modes == 1:
        yield (total,)
    else:
        for first in range(total + 1):
            for tail in compositions(total - first, modes - 1):
                yield (first, *tail)


class FockBasis:
    """All occupations with TOTAL photon number <= max_photons (not a per-mode cutoff)."""
    def __init__(self, modes, max_photons, *, precision=None):
        self.modes = integer(modes, "modes", 1)
        self.max_photons = integer(max_photons, "max_photons")
        self.precision = precision or Precision()
        self.dimension = comb(modes + max_photons, max_photons)
        self.precision.guard(self.dimension)
        self.states = tuple(s for n in range(max_photons + 1) for s in compositions(n, modes))
        self.index = {s: i for i, s in enumerate(self.states)}
        self.occupations = np.array(self.states, dtype=int)
        self.occupations.flags.writeable = False

    def __eq__(self, other):
        return isinstance(other, FockBasis) and (self.modes, self.max_photons) == (other.modes, other.max_photons)

    def validate_occupation(self, occupation):
        occ = tuple(integer(n, "occupation") for n in occupation)
        if len(occ) != self.modes or sum(occ) > self.max_photons:
            raise ValueError("Occupation exceeds the specified modes or total-photon cutoff")
        return occ

    def __repr__(self):
        return f"FockBasis(modes={self.modes}, max_photons={self.max_photons}, dimension={self.dimension})"


class FockState:
    def __init__(self, basis, state, *, subnormalized=False):
        self.basis = basis
        self.rho = density(state, basis.dimension, subnormalized=subnormalized, atol=basis.precision.atol)

    @classmethod
    def ket(cls, basis, occupation):
        v = np.zeros(basis.dimension, complex)
        v[basis.index[basis.validate_occupation(occupation)]] = 1
        return cls(basis, v)

    @classmethod
    def amplitudes(cls, basis, amplitudes, *, subnormalized=False):
        v = np.zeros(basis.dimension, complex)
        for occ, amplitude in amplitudes.items():
            v[basis.index[basis.validate_occupation(occ)]] = amplitude
        return cls(basis, v, subnormalized=subnormalized)

    @classmethod
    def mixture(cls, basis, probabilities, *, subnormalized=False):
        diag = np.zeros(basis.dimension)
        for occ, p in probabilities.items():
            diag[basis.index[basis.validate_occupation(occ)]] = probability(p)
        return cls(basis, np.diag(diag), subnormalized=subnormalized)

    @property
    def trace(self):
        return float(np.trace(self.rho).real)

    def probabilities(self):
        return {s: float(max(0, self.rho[i, i].real)) for i, s in enumerate(self.basis.states)}

    def conditional(self):
        return FockState(self.basis, Branch(self.rho).conditional())

    def diagnostics(self):
        return {**diagnostics(self.rho), "dimension": self.basis.dimension,
                "total_photon_cutoff": self.basis.max_photons,
                "boundary_probability": float(sum(self.rho[i, i].real for i, s in enumerate(self.basis.states)
                                                  if sum(s) == self.basis.max_photons))}

    def postselect(self, predicate):
        """Ideal nondestructive projector. Retains the unnormalized success branch."""
        keep = np.array([bool(predicate(s)) for s in self.basis.states])
        return FockState(self.basis, self.rho * np.outer(keep, keep), subnormalized=True)


@dataclass
class SourceResult:
    state: FockState
    omitted_probability: float


def independent_sources(basis, distributions):
    """Per-mode explicit number distributions, including vacuum/multiphoton contamination.

    The retained state is NOT renormalized if the TOTAL cutoff removes configurations.
    Input distributions themselves must sum to one. Photons are indistinguishable.
    """
    if len(distributions) != basis.modes:
        raise ValueError("One number distribution is required per mode")
    arrays = []
    for dist in distributions:
        p = np.array([probability(v) for v in dist])
        if not len(p) or abs(p.sum() - 1) > basis.precision.atol:
            raise ValueError("Each source number distribution must sum to one")
        arrays.append(p)
    probs = {}
    for occ in basis.states:
        probs[occ] = float(np.prod([p[n] if n < len(p) else 0 for p, n in zip(arrays, occ)]))
    state = FockState.mixture(basis, probs, subnormalized=True)
    return SourceResult(state, max(0.0, 1-state.trace))


def lift_unitary(single_particle, basis):
    """Normalized creation-operator recurrence, including all bunching sectors.

    Convention: a_j^dagger -> sum_i U[i,j] a_i^dagger.
    """
    u = unitary(single_particle, basis.precision.atol)
    if u.shape != (basis.modes, basis.modes):
        raise ValueError("Unitary and mode counts differ")
    result = np.zeros((basis.dimension, basis.dimension), complex)
    for col, occ in enumerate(basis.states):
        amplitudes = {(0,) * basis.modes: 1.0 + 0j}
        for mode, count in enumerate(occ):
            for k in range(1, count + 1):
                updated = {}
                for state, amp in amplitudes.items():
                    for out in range(basis.modes):
                        if u[out, mode] == 0:
                            continue
                        nxt = list(state)
                        nxt[out] += 1
                        key = tuple(nxt)
                        updated[key] = updated.get(key, 0j) + amp * u[out, mode] * np.sqrt(nxt[out]/k)
                amplitudes = updated
        for state, amp in amplitudes.items():
            result[basis.index[state], col] = amp
    return result


def loss_operators(basis, mode, transmission):
    mode = integer(mode, "mode")
    if mode >= basis.modes:
        raise ValueError("Mode out of range")
    eta = probability(transmission, "transmission")
    basis.precision.guard_kraus(basis.dimension,basis.dimension,basis.max_photons+1)
    operators = []
    for lost in range(basis.max_photons + 1):
        k = np.zeros((basis.dimension, basis.dimension), complex)
        for col, occ in enumerate(basis.states):
            n = occ[mode]
            if n >= lost:
                dest = list(occ)
                dest[mode] -= lost
                k[basis.index[tuple(dest)], col] = np.sqrt(comb(n, lost) * (1-eta)**lost * eta**(n-lost))
        operators.append(k)
    return operators


def apply_loss(rho, basis, mode, transmission):
    """Sparse-index evaluation of the exact pure-loss map without dense Kraus storage."""
    eta = probability(transmission,"transmission")
    if integer(mode,"mode") >= basis.modes:
        raise ValueError("Mode out of range")
    output = np.zeros_like(rho)
    for lost in range(basis.max_photons+1):
        source, destination, coefficients = [], [], []
        for col, occupation in enumerate(basis.states):
            n=occupation[mode]
            if n < lost:
                continue
            dest=list(occupation)
            dest[mode]-=lost
            source.append(col)
            destination.append(basis.index[tuple(dest)])
            coefficients.append(np.sqrt(comb(n,lost)*(1-eta)**lost*eta**(n-lost)))
        c=np.asarray(coefficients)
        output[np.ix_(destination,destination)] += rho[np.ix_(source,source)] * np.outer(c,c)
    return output


def phase_kernel(basis, covariance, mean=None):
    cov = finite_array(covariance, 2)
    if cov.shape != (basis.modes, basis.modes) or np.max(abs(cov.imag)) > basis.precision.atol:
        raise ValueError("Phase covariance must be a real modes-by-modes matrix (radians squared)")
    cov = cov.real
    if not np.allclose(cov, cov.T, atol=basis.precision.atol, rtol=0) or np.linalg.eigvalsh(cov).min() < -basis.precision.atol:
        raise ValueError("Phase covariance must be symmetric positive semidefinite")
    mu = np.zeros(basis.modes) if mean is None else finite_array(mean, 1)
    if mu.shape != (basis.modes,) or np.max(abs(mu.imag)) > basis.precision.atol:
        raise ValueError("Mean phases must be a real vector in radians")
    delta = basis.occupations[:, None, :] - basis.occupations[None, :, :]
    return np.exp(1j * (delta @ mu.real) - 0.5 * np.einsum("...i,ij,...j->...", delta, cov, delta))


@dataclass
class OpticalResult:
    state: FockState
    trace_history: tuple
    model: str = "complex128 passive Fock density matrix; vacuum-environment loss; Gaussian phase noise"
    truncation_history: tuple = ()
    transfer_residuals: tuple = ()

    @property
    def bath_omitted_probability(self):
        return float(sum(item["omitted_weight"] for item in self.truncation_history))

    @property
    def numerical_trace_error(self):
        return abs(self.state.trace - (self.trace_history[0]-self.bath_omitted_probability))

    def probabilities(self):
        return self.state.probabilities()


class Circuit:
    def __init__(self, modes):
        self.modes = integer(modes, "modes", 1)
        self.steps = []
        self.transfer_residuals = []

    def _mode(self, mode):
        if integer(mode, "mode") >= self.modes:
            raise ValueError("Mode out of range")
        return int(mode)

    def unitary(self, matrix, modes=None):
        u = unitary(matrix)
        targets = tuple(range(self.modes)) if modes is None else tuple(self._mode(q) for q in modes)
        if len(set(targets)) != len(targets) or u.shape != (len(targets),) * 2:
            raise ValueError("Unitary shape or target modes are invalid")
        full = np.eye(self.modes, dtype=complex)
        full[np.ix_(targets, targets)] = u
        self.steps.append(("unitary", full))
        return self

    def bs(self, first, second, *, transmission=0.5, phase=0.0):
        """Power transmission T; matrix [[sqrt(T),-e^-iφ sqrt(1-T)],[e^iφ sqrt(1-T),sqrt(T)]]."""
        t = probability(transmission, "transmission")
        if not np.isfinite(phase) or not np.isreal(phase):
            raise ValueError("Beam-splitter phase must be finite and real")
        u = [[np.sqrt(t), -np.exp(-1j*phase)*np.sqrt(1-t)],
             [np.exp(1j*phase)*np.sqrt(1-t), np.sqrt(t)]]
        return self.unitary(u, (first, second))

    def transfer(self, matrix, modes=None, *, atol=1e-10):
        """Passive, possibly lossy field transfer A=U diag(s) Vh.

        Apply Vh, independent vacuum losses s**2, then U. Unlike applying A
        as a nonunitary ket gate, this retains lost-photon sectors and the
        environmental which-path information. Only square contractions are
        supported. Singular-value excess within atol is treated as roundoff;
        the actual field-matrix residual is recorded in the run result.
        """
        if not np.isfinite(atol) or not 0 < atol < .01:
            raise ValueError("Transfer atol must be finite and in (0, .01)")
        a = finite_array(matrix, 2)
        targets = tuple(range(self.modes)) if modes is None else tuple(self._mode(m) for m in modes)
        if not targets or len(set(targets)) != len(targets) or a.shape != (len(targets),)*2:
            raise ValueError("Transfer shape or target modes are invalid")
        u, s, vh = np.linalg.svd(a)
        if s.max() > 1+atol:
            raise ValueError("Passive field transfer must be a contraction; gain requires an active-noise model")
        retained = np.minimum(s, 1.)
        residual = float(np.linalg.norm((u*retained)@vh-a))
        self.transfer_residuals.append({"modes": targets, "field_matrix_residual": residual,
                                        "largest_input_singular_value": float(s.max())})
        self.unitary(vh, targets)
        for mode, amplitude in zip(targets, retained):
            self.loss(mode, float(amplitude*amplitude))
        return self.unitary(u, targets)

    def phase(self, mode, radians):
        if not np.isfinite(radians) or not np.isreal(radians):
            raise ValueError("Phase must be finite and real")
        return self.unitary([[np.exp(1j*radians)]], (mode,))

    def loss(self, mode, transmission):
        self.steps.append(("loss", (self._mode(mode), probability(transmission, "transmission"))))
        return self

    def thermal_loss(self, mode, transmission, mean_photons, *, bath_cutoff=None, tail_tolerance=1e-12):
        """Mix with an independent thermal bath, retaining its omitted weight.

        The total system cutoff grows by bath_cutoff at this step. If omitted,
        the bath cutoff is chosen from its geometric tail <= tail_tolerance.
        This is a thermal attenuator, not a detector dark-count model.
        """
        from .thermal import ThermalBath
        bath = ThermalBath.create(mean_photons, bath_cutoff, tail_tolerance)
        self.steps.append(("thermal_loss", (self._mode(mode), probability(transmission, "transmission"), bath)))
        return self

    def phase_noise(self, covariance, mean=None):
        cov = finite_array(covariance, 2)
        mu = np.zeros(self.modes) if mean is None else finite_array(mean, 1)
        phase_kernel(FockBasis(self.modes, 0), cov, mu)  # validates physical parameters
        self.steps.append(("phase_noise", (cov.copy(), mu.copy())))
        return self

    def single_particle_unitary(self):
        result = np.eye(self.modes, dtype=complex)
        for kind, data in self.steps:
            if kind != "unitary":
                raise ValueError("Circuit contains noise; it cannot be exported as a lossless unitary")
            result = data @ result
        return result

    def run(self, state, *, precision=None):
        """Run a FockState, or an occupation such as [1, 1] with automatic cutoff.

        Occupation shorthand uses exactly sum(occupation) initial photons.
        Mixed states and superpositions use the explicit FockState constructors.
        """
        if not isinstance(state, FockState):
            occupation = tuple(integer(n, "occupation") for n in state)
            if len(occupation) != self.modes:
                raise ValueError("One occupation is required per circuit mode")
            state = FockState.ket(FockBasis(self.modes, sum(occupation), precision=precision), occupation)
        elif precision is not None:
            raise ValueError("For FockState inputs, set precision on its FockBasis")
        basis = state.basis
        if basis.modes != self.modes:
            raise ValueError("Input and circuit mode counts differ")
        rho = state.rho.copy()
        history = [float(np.trace(rho).real)]
        truncations = []
        for step_index, (kind, data) in enumerate(self.steps):
            if kind == "unitary":
                u = lift_unitary(data, basis)
                rho = u @ rho @ u.conj().T
            elif kind == "loss":
                rho = apply_loss(rho, basis, *data)
            elif kind == "phase_noise":
                rho *= phase_kernel(basis, *data)
            elif kind == "thermal_loss":
                from .thermal import apply_thermal_loss
                mode, eta, bath = data
                tail = 0. if eta == 1 else bath.omitted_probability
                truncations.append({"step": step_index, "bath_cutoff": bath.cutoff,
                                    "bath_tail_probability": tail,
                                    "omitted_weight": float(np.trace(rho).real)*tail})
                rho, basis = apply_thermal_loss(rho, basis, mode, eta, bath)
            else:
                raise ValueError(f"Unknown optical operation: {kind}")
            history.append(float(np.trace(rho).real))
        model = OpticalResult.model
        if truncations:
            model += "; thermal beam-splitter baths with explicit unnormalized truncation"
        return OpticalResult(FockState(basis, rho, subnormalized=True), tuple(history), model,
                             tuple(truncations), tuple(self.transfer_residuals))

    def run_sparse(self, state, *, budget=None):
        """Pure lossless Fock amplitudes without a lifted dense matrix."""
        from .scalable import run_sparse
        return run_sparse(self, state, budget=budget)

    def sample(self, state, *, shots=1000, seed=0, budget=None, detectors=None):
        """Explicit Monte Carlo loss/phase trajectories; see scalable.py."""
        from .scalable import sample_trajectories
        return sample_trajectories(self, state, shots=shots, seed=seed, budget=budget, detectors=detectors)

    def channel(self, basis):
        """Explicit full Kraus map for SMALL optical instruments; count is budgeted."""
        if basis.modes != self.modes:
            raise ValueError("Basis and circuit differ")
        input_basis = basis
        trace_preserving = True
        operators = [np.eye(basis.dimension, dtype=complex)]
        for kind, data in self.steps:
            if kind == "unitary":
                layer = [lift_unitary(data, basis)]
            elif kind == "loss":
                layer = loss_operators(basis, *data)
            elif kind == "phase_noise":
                values, vectors = np.linalg.eigh(phase_kernel(basis, *data))
                basis.precision.guard_kraus(basis.dimension,basis.dimension,int(np.count_nonzero(values>0)))
                layer = [np.diag(np.sqrt(max(0, x)) * vectors[:, i]) for i, x in enumerate(values) if x > 0]
            elif kind == "thermal_loss":
                from .thermal import thermal_loss_operators
                layer, basis = thermal_loss_operators(basis, *data)
                if data[1] != 1 and data[2].mean_photons > 0:
                    trace_preserving = False
            else:
                raise ValueError(f"Unknown optical operation: {kind}")
            layer = [k for k in layer if np.any(k)]
            if len(layer) * len(operators) > basis.precision.max_kraus:
                raise ResourceLimitError("Explicit optical Kraus expansion exceeds max_kraus; use Circuit.run")
            basis.precision.guard_kraus(basis.dimension,input_basis.dimension,len(layer)*len(operators))
            operators = [b @ a for b in layer for a in operators]
        channel = KrausChannel(operators, trace_preserving=trace_preserving, precision=basis.precision, name="optical")
        channel.input_basis, channel.output_basis = input_basis, basis
        return channel


class DualRail:
    """|0> = |1,0>, |1> = |0,1>. Qubit zero is the most significant bit."""
    def __init__(self, n_qubits, *, basis=None, precision=None):
        self.n_qubits = integer(n_qubits, "n_qubits", 1)
        self.basis = basis or FockBasis(2*n_qubits, n_qubits, precision=precision)
        if self.basis.modes != 2*n_qubits or self.basis.max_photons < n_qubits:
            raise ValueError("Dual-rail basis needs exactly 2*n modes and cutoff >= n")

    @cached_property
    def isometry(self):
        v = np.zeros((self.basis.dimension, 2**self.n_qubits), complex)
        for i in range(2**self.n_qubits):
            occ = []
            for q in range(self.n_qubits):
                bit = (i >> (self.n_qubits-q-1)) & 1
                occ.extend((1-bit, bit))
            v[self.basis.index[tuple(occ)], i] = 1
        return v

    def encode(self, state):
        rho = density(state, 2**self.n_qubits, subnormalized=True)
        return FockState(self.basis, self.isometry @ rho @ self.isometry.conj().T, subnormalized=True)

    def decode(self, state):
        if state.basis != self.basis:
            # Thermal baths enlarge the number cutoff without changing rail
            # labels. Rebuild only the computational embedding in that basis.
            v = DualRail(self.n_qubits, basis=state.basis).isometry
        else:
            v = self.isometry
        return Branch(v.conj().T @ state.rho @ v)

    def effective_channel(self, circuit):
        """TNI computational branch; loss AND bunching remain in the failure probability."""
        channel = circuit.channel(self.basis)
        v = self.isometry
        output_v = DualRail(self.n_qubits, basis=channel.output_basis).isometry
        return KrausChannel([output_v.conj().T @ k @ v for k in channel.operators], trace_preserving=False,
                            precision=self.basis.precision, name="dual-rail success")
