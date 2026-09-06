"""Lossless Perceval circuit interchange with explicit mode conventions."""
import numpy as np
from ..optics import Circuit


def from_perceval(circuit):
    import perceval as pcvl
    if not isinstance(circuit, pcvl.ACircuit):
        raise TypeError("Expected a Perceval ACircuit. Processor source/noise/herald settings are not imported; specify them in PLQ")
    matrix = np.array(circuit.compute_unitary(use_symbolic=False), dtype=complex)
    if matrix.shape != (circuit.m,)*2:
        raise ValueError("Polarization-resolved circuits require explicit expansion into PLQ modes")
    return Circuit(circuit.m).unitary(matrix)


def to_perceval(circuit):
    import perceval as pcvl
    u = circuit.single_particle_unitary()  # noisy circuits fail here; nothing is silently dropped
    return pcvl.Circuit(circuit.modes).add(0, pcvl.Unitary(pcvl.Matrix(u)))


def reference_probabilities(circuit, occupation, *, backend="SLOS"):
    import perceval as pcvl
    optical = to_perceval(circuit) if isinstance(circuit,Circuit) else circuit
    if not isinstance(optical,pcvl.ACircuit):
        raise TypeError("Expected a unitary optical circuit")
    from ..numerics import integer
    occupation = tuple(integer(n,"occupation") for n in occupation)
    if len(occupation) != optical.m:
        raise ValueError("Occupation and mode count differ")
    engine = pcvl.BackendFactory.get_backend(backend)
    engine.set_circuit(optical)
    engine.set_input_state(pcvl.BasicState(list(occupation)))
    return {tuple(int(n) for n in state):float(p) for state,p in engine.prob_distribution().items()}
