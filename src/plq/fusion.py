"""Unboosted dual-rail linear-optical Bell measurement, with no fictitious deterministic entangler."""
from itertools import product
import numpy as np
from .optics import Circuit, DualRail
from .detectors import Detector, detection_probabilities
from .channels import KrausChannel


def bell_analyzer_circuit():
    return Circuit(4).bs(0,2).bs(1,3)


def bell_label(counts):
    counts = tuple(counts)
    if counts in ((0,0,1,1), (1,1,0,0)):
        return "psi_plus"
    if counts in ((0,1,1,0), (1,0,0,1)):
        return "psi_minus"
    return "failure"


def bell_measurement(state, *, detectors=None, transmission=1.0):
    """Input: normalized 2-qubit density/vector. Labels can be false heralds with noise.

    Four Bell states uniformly supplied give success probability 1/2 ideally.
    This final measurement consumes both qubits. For resource-state fusion, use
    the same beam splitters and detectors with herald() on a larger Fock state.
    """
    rails = DualRail(2)
    circuit = bell_analyzer_circuit()
    for mode in range(4):
        circuit.loss(mode, transmission)
    detectors = tuple(detectors) if detectors is not None else (Detector(saturation=3),)*4
    counts = detection_probabilities(circuit.run(rails.encode(state)).state, detectors=detectors)
    labels = {"psi_plus":0., "psi_minus":0., "failure":0.}
    for pattern,p in counts.items():
        labels[bell_label(pattern)] += p
    return labels


def bell_instruments(*, detectors=None):
    """Three CP maps 4 -> 1 for a destructive Bell analyzer (no propagation loss here).

    Inefficiency/dark counts are number-diagonal detector effects. Individual
    maps are trace nonincreasing; their union is a complete instrument.
    """
    rails = DualRail(2)
    detectors = tuple(detectors) if detectors is not None else (Detector(saturation=3),)*4
    if len(detectors) != 4:
        raise ValueError("Four detectors are required")
    u = bell_analyzer_circuit().channel(rails.basis).operators[0] @ rails.isometry
    responses = [[d.response(n) for n in range(3)] for d in detectors]
    weights = {label:np.zeros(rails.basis.dimension) for label in ("psi_plus","psi_minus","failure")}
    for counts in product(*(d.outcomes for d in detectors)):
        label = bell_label(counts)
        for i,occ in enumerate(rails.basis.states):
            weights[label][i] += np.prod([responses[j][n][counts[j]] for j,n in enumerate(occ)])
    return {label:KrausChannel([np.sqrt(w)*u[i:i+1] for i,w in enumerate(values)],
                               trace_preserving=False, name=label) for label,values in weights.items()}
