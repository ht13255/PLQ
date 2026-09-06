"""Reproducible HOM experiments with explicit source and detector hypotheses."""
import numpy as np
from .detectors import Detector
from .numerics import Precision, probability
from .optics import Circuit
from .sources import number_distribution, number_moments
from .wavepackets import wavepacket_sources


def source_hom(distributions, indistinguishability, *, multiphoton_model,
               beamsplitter_transmission=.5, transmissions=(1., 1.),
               detectors=None, precision=None):
    """Simulate two independent imperfect sources and a distinguishable reference.

    indistinguishability is M=|<phi_0|phi_1>|**2 for the SINGLE-photon
    components, not an amplitude or a measured raw HOM visibility. The
    multiphoton hypothesis is mandatory. All supported number sectors are
    propagated through optics and threshold detection; no weak-source or
    low-detector-efficiency approximation is used. The reference changes the
    signal overlap to zero while preserving all other settings.
    """
    precision = precision or Precision()
    arrays = tuple(number_distribution(p, atol=precision.atol) for p in distributions)
    if len(arrays) != 2:
        raise ValueError("source_hom requires two number distributions")
    m = probability(indistinguishability, "indistinguishability")
    t = probability(beamsplitter_transmission, "beamsplitter_transmission")
    etas = tuple(probability(v, "transmission") for v in transmissions)
    if len(etas) != 2:
        raise ValueError("Two output transmissions are required")
    detectors = tuple(detectors) if detectors is not None else (Detector(threshold=True),)*2
    if len(detectors) != 2 or any(not isinstance(d, Detector) or not d.threshold for d in detectors):
        raise ValueError("source_hom requires two threshold detectors")
    circuit = Circuit(2).bs(0, 1, transmission=t).loss(0, etas[0]).loss(1, etas[1])

    def run(overlap_squared):
        amplitude = np.sqrt(overlap_squared)
        source = wavepacket_sources(2, [0, 1], arrays, [[1, amplitude], [amplitude, 1]],
                                   multiphoton_model=multiphoton_model, precision=precision)
        output = source.through(circuit)
        counts = output.detection_probabilities(detectors)
        spatial = output.spatial_probabilities()
        return {"coincidence_probability": counts[(1, 1)],
                "singles_probabilities": [sum(p for c, p in counts.items() if c[j]) for j in range(2)],
                "intensity_cross_moment": sum(n[0]*n[1]*p for n, p in spatial.items()),
                "mean_output_photons": [sum(n[j]*p for n, p in spatial.items()) for j in range(2)],
                "total_probability": sum(counts.values()),
                "source_omitted_probability": source.source_omitted_probability,
                "gram_factorization_residual": source.gram_factorization_residual,
                "hilbert_dimension": source.state.basis.dimension}

    parallel, distinguishable = run(m), run(0.)
    reference = distinguishable["coincidence_probability"]
    return {"parallel": parallel, "distinguishable": distinguishable,
            "click_visibility": None if reference == 0 else
                1-parallel["coincidence_probability"]/reference,
            "source_moments": [number_moments(p) for p in arrays],
            "single_photon_indistinguishability": m,
            "multiphoton_model": multiphoton_model,
            "model": "independent number-diagonal sources; explicit internal modes; threshold clicks",
            "assumptions": ["Each single-photon component is a pure wavepacket.",
                            "The declared number support is exact; no higher photon numbers are inferred.",
                            "Pulse-to-pulse correlations and detector recovery are not modeled.",
                            "The distinguishable reference is a separate two-input experiment, not a pulsed histogram normalization."]}
