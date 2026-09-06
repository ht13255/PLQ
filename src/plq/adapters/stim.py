"""Large Clifford/QEC circuits supplied by the user, including custom .stim files."""
from dataclasses import dataclass, asdict
import numpy as np
from ..simulation import wilson_interval
from ..numerics import integer


@dataclass
class StimResult:
    shots: int
    failures: int
    logical_error_rate: float
    wilson_95: tuple
    detectors: int
    observables: int
    seed: int
    versions: dict
    model: str = "user-supplied Stim circuit and detector error model; any-observable failure per shot"

    def to_dict(self):
        return asdict(self)


def simulate_stim(circuit, *, shots=10000, seed=0, decoder=None, correlated_matching=False):
    """Decode with PyMatching, or a user object implementing decode_batch(detections).

    Stim's Clifford/Pauli/measurement model is the simulated model. Coherent
    optical amplitudes and physical photon loss are not inferred from the text.
    Unsupported/non-graphlike DEM decompositions raise; error terms are not dropped.
    """
    import stim
    shots,seed = integer(shots,"shots",1),integer(seed,"seed")
    if isinstance(circuit,str):
        circuit = stim.Circuit(circuit)
    if not isinstance(circuit,stim.Circuit):
        raise TypeError("Expected stim.Circuit or .stim source text")
    if circuit.num_observables == 0:
        raise ValueError("At least one OBSERVABLE_INCLUDE is needed to measure logical failure")
    versions = {"stim":stim.__version__, "numpy":np.__version__}
    if decoder is None:
        import pymatching
        dem = circuit.detector_error_model(decompose_errors=True)
        decoder = pymatching.Matching.from_detector_error_model(dem, enable_correlations=correlated_matching)
        versions["pymatching"] = pymatching.__version__
        decode = lambda events: decoder.decode_batch(events, enable_correlations=correlated_matching)
    else:
        decode = decoder.decode_batch
    detections, truth = circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    predictions = np.asarray(decode(detections))
    if predictions.shape != truth.shape or not np.all(np.isin(predictions,(0,1))):
        raise ValueError("Decoder predictions must be binary and have shape (shots, num_observables)")
    failures = int(np.count_nonzero(np.any(predictions != truth,axis=1)))
    return StimResult(shots,failures,failures/shots,wilson_interval(failures,shots),
                      circuit.num_detectors,circuit.num_observables,seed,versions)
