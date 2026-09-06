"""A physical six-rail teleportation instrument with explicit classical events.

This is a component bridge, not a fault-tolerant photonic computer compiler.
Accepted detector events are not proof that the output photon survived.
"""
from dataclasses import dataclass, asdict
import numpy as np
from .channels import KrausChannel, PAULI
from .numerics import density, probability
from .optics import Circuit, FockBasis, FockState, DualRail
from .detectors import Detector, herald
from .fusion import bell_label

PATTERNS = ((0, 0, 1, 1), (1, 1, 0, 0), (0, 1, 1, 0), (1, 0, 0, 1))


@dataclass(frozen=True)
class FeedForward:
    detection_seconds: float = 0.
    processing_seconds: float = 0.
    switching_seconds: float = 0.
    buffer_seconds: float = 0.
    buffer_loss_db_per_second: float = 0.
    switch_transmission: float = 1.
    phase_variance_per_second: float = 0.

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isscalar(value) or not np.isreal(value) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite, real and nonnegative")
        probability(self.switch_transmission, "switch_transmission")

    @property
    def latency_seconds(self):
        return self.detection_seconds+self.processing_seconds+self.switching_seconds

    @property
    def meets_deadline(self):
        return self.latency_seconds <= self.buffer_seconds

    @property
    def transmission(self):
        return float(self.switch_transmission*10**(-self.buffer_loss_db_per_second*self.buffer_seconds/10))


def _tomography_channel(outputs, name):
    """Reconstruct a linear CP map from four spanning input density matrices."""
    a, b, plus, plus_i = outputs
    c, d = plus-(a+b)/2, plus_i-(a+b)/2
    choi = np.block([[a, c+1j*d], [c-1j*d, b]])
    hermitian_residual = float(np.linalg.norm(choi-choi.conj().T))
    eigenvalues, eigenvectors = np.linalg.eigh((choi+choi.conj().T)/2)
    if eigenvalues.min() < -1e-10 or hermitian_residual > 1e-10:
        raise ValueError("Optical process reconstruction is not CP within complex128 tolerance")
    operators = [np.sqrt(v)*eigenvectors[:, i].reshape((len(a), 2), order="F")
                 for i, v in enumerate(eigenvalues) if v > 0]
    if not operators:
        operators = [np.zeros((len(a), 2), complex)]
    channel = KrausChannel(operators, trace_preserving=False, name=name)
    # Report the roundoff-only PSD projection explicitly; positive eigenvalues
    # are never cut at an arbitrary rank tolerance.
    channel.reconstruction = {"negative_eigenvalue_weight": float(-eigenvalues[eigenvalues < 0].sum()),
                              "hermiticity_residual": hermitian_residual}
    return channel


@dataclass
class TeleportationInstrument:
    events: dict
    output_basis: FockBasis
    schedule: FeedForward
    rejected_effect: np.ndarray
    late_effect: np.ndarray
    metadata: dict

    def apply(self, state):
        rho = density(state, 2)
        accepted = {str(pattern): channel.apply(rho) for pattern, channel in self.events.items()}
        output = sum((branch.rho for branch in accepted.values()),
                     np.zeros((self.output_basis.dimension,)*2, complex))
        computational = DualRail(1, basis=self.output_basis).decode(
            FockState(self.output_basis, output, subnormalized=True))
        p = float(np.trace(output).real)
        return {"accepted_events": accepted, "accepted_state": FockState(self.output_basis, output, subnormalized=True),
                "accepted_probability": p, "computational_branch": computational,
                "accepted_leakage_probability": max(0., p-computational.probability),
                "rejected_probability": float(np.trace(self.rejected_effect@rho).real),
                "late_probability": float(np.trace(self.late_effect@rho).real)}

    def computational_channel(self):
        """Diagnostic TNI channel; membership is NOT a nondestructively observed flag.

        Can be composed with LogicalQPU.local_channel for small-code analysis.
        It conditions on an unmeasured subspace, so preserve its absolute trace.
        """
        v = DualRail(1, basis=self.output_basis).isometry
        return KrausChannel([v.conj().T@k for ch in self.events.values() for k in ch.operators],
                            trace_preserving=False, name="teleportation computational diagnostic")

    def flagged_channel(self):
        """CPTP qubit -> [event-specific Fock blocks, rejected, late].

        Vacuum/leakage stays INSIDE accepted blocks; only actually known
        classical rejection/deadline events are flags. No Pauli twirl.
        """
        width = self.output_basis.dimension
        size = len(self.events)*width+2
        operators = []
        for index, channel in enumerate(self.events.values()):
            for k in channel.operators:
                op = np.zeros((size, 2), complex)
                op[index*width:(index+1)*width] = k
                operators.append(op)
        for row, effect in ((size-2, self.rejected_effect), (size-1, self.late_effect)):
            values, vectors = np.linalg.eigh((effect+effect.conj().T)/2)
            for i, value in enumerate(values):
                if value > 0:
                    op = np.zeros((size, 2), complex)
                    op[row] = np.sqrt(value)*vectors[:, i].conj()
                    operators.append(op)
        return KrausChannel(operators, name="event-resolved optical teleportation")


def teleportation_instrument(*, resource=None, transmissions=(1.,)*6, detectors=None,
                             beamsplitter_transmission=.5, schedule=None, output_phase=0.,
                             analyzer_transfer=None):
    """Input rails 0,1; Bell resource 2,3/4,5; measure 0..3; retain 4,5.

    Resource defaults to |Phi+>. An explicit four-mode FockState may include
    vacuum and up to two photons (including multiphoton contaminants).
    All photons share one internal wavepacket. analyzer_transfer is an optional
    calibrated passive 4x4 field matrix replacing the two beam splitters.
    Detector responses are independent per invocation; this does not model
    detector recovery across pulses. Timing parameters are user assumptions.
    """
    if analyzer_transfer is not None and beamsplitter_transmission != .5:
        raise ValueError("analyzer_transfer replaces the beam splitters; omit beamsplitter_transmission")
    schedule = schedule or FeedForward()
    detectors = tuple(detectors) if detectors is not None else (Detector(saturation=4),)*4
    if len(detectors) != 4 or any(not d.threshold and d.saturation < 2 for d in detectors):
        raise ValueError("Four detectors required; PNR saturation must be >=2 to distinguish one from overflow")
    transmissions = tuple(probability(v, "transmission") for v in transmissions)
    if len(transmissions) != 6:
        raise ValueError("Six rail transmissions are required")
    if resource is None:
        resource = DualRail(2).encode(np.array([1, 0, 0, 1])/np.sqrt(2))
    if not isinstance(resource, FockState) or resource.basis.modes != 4 or resource.basis.max_photons > 2:
        raise ValueError("Resource must be a four-mode FockState with total cutoff <=2")
    if abs(resource.trace-1) > 1e-10:
        raise ValueError("Resource must be normalized; include vacuum rather than postselecting source success")
    basis = FockBasis(6, 1+resource.basis.max_photons)
    output_basis = FockBasis(2, basis.max_photons)
    circuit = Circuit(6)
    for mode, eta in enumerate(transmissions):
        if eta != 1:
            circuit.loss(mode, eta)
    if analyzer_transfer is None:
        circuit.bs(0, 2, transmission=beamsplitter_transmission).bs(1, 3, transmission=beamsplitter_transmission)
    else:
        circuit.transfer(analyzer_transfer, modes=(0, 1, 2, 3))
    # Fixed buffering is applied even if control is late; late events are
    # explicitly discarded, rather than assuming an instantaneous correction.
    if schedule.transmission != 1:
        circuit.loss(4, schedule.transmission).loss(5, schedule.transmission)
    if schedule.phase_variance_per_second:
        cov = np.zeros((6, 6))
        cov[5, 5] = schedule.phase_variance_per_second*schedule.buffer_seconds
        circuit.phase_noise(cov)
    circuit.phase(5, output_phase)
    corrections = {}
    for pattern in PATTERNS:
        correction = Circuit(2).unitary(PAULI["X"])
        if bell_label(pattern) == "psi_minus":
            correction.unitary(PAULI["Z"])
        from .optics import lift_unitary
        corrections[pattern] = lift_unitary(correction.single_particle_unitary(), output_basis)
    states = [np.array([1, 0]), np.array([0, 1]), np.array([1, 1])/np.sqrt(2), np.array([1, 1j])/np.sqrt(2)]
    outputs = {p: [] for p in PATTERNS}
    indices = [basis.index[(1-bit, bit, *occ)] for bit in range(2) for occ in resource.basis.states]
    for psi in states:
        rho = np.zeros((basis.dimension,)*2, complex)
        rho[np.ix_(indices, indices)] = np.kron(np.outer(psi, psi.conj()), resource.rho)
        result = circuit.run(FockState(basis, rho)).state
        for pattern in PATTERNS:
            remaining = herald(result, (0, 1, 2, 3), pattern, detectors).remaining_state
            u = corrections[pattern]
            outputs[pattern].append(u@remaining.rho@u.conj().T)
    events = {p: _tomography_channel(values, f"teleportation {p}") for p, values in outputs.items()}
    accepted_effect = sum(k.conj().T@k for ch in events.values() for k in ch.operators)
    rejected_effect = np.eye(2)-accepted_effect
    if np.linalg.eigvalsh(rejected_effect).min() < -1e-10:
        raise ValueError("Accepted detector instrument increases probability")
    late_effect = np.zeros((2, 2), complex)
    reconstruction = {str(p): ch.reconstruction for p, ch in events.items()}
    if not schedule.meets_deadline:
        late_effect = accepted_effect
        events = {p: KrausChannel([np.zeros((output_basis.dimension, 2))], trace_preserving=False) for p in PATTERNS}
    return TeleportationInstrument(events, output_basis, schedule, rejected_effect, late_effect,
        {"model": "six-rail indistinguishable Fock teleportation; detector-conditioned CP maps; physical Pauli feedforward",
         "schedule": asdict(schedule), "latency_seconds": schedule.latency_seconds,
         "meets_deadline": schedule.meets_deadline, "buffer_and_switch_transmission": schedule.transmission,
         "transmissions": transmissions, "detectors": [asdict(d) for d in detectors],
         "resource_trace": resource.trace, "resource_cutoff": resource.basis.max_photons,
         "transfer_residuals": circuit.transfer_residuals, "reconstruction": reconstruction,
         "limitations": ["Resource preparation circuit is not synthesized; explicit resource state is required.",
             "Common internal wavepacket; no spectral correlations or inter-pulse detector memory.",
             "Computational projection is diagnostic; output loss is not automatically heralded.",
             "This primitive does not establish a fault-tolerant schedule or a QEC threshold."]})
