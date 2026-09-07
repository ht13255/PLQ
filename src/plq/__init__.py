"""PLQ: transparent photonic logical-qubit simulation."""
__version__ = "0.4.0"

from .numerics import Precision, ResourceLimitError, diagnostics
from .channels import Branch, KrausChannel, PAULI, pauli_channel, erasure_channel
from .optics import FockBasis, FockState, Circuit, DualRail, independent_sources
from .detectors import Detector, HeraldResult, detection_probabilities, herald
from .wavepackets import (WavepacketState, wavepacket_input, gaussian_gram,
                         mixed_wavepacket_input, wavepacket_sources)
from .sources import number_distribution_from_moments, number_moments
from .experiments import source_hom
from .qec import (CodeSpace, StabilizerCode, Decoder, DecodeFailure, MinimumWeightDecoder, ErasureDecoder,
                  css_code, repetition_code, five_qubit_code, steane_code, shor_code, get_code, register_code)
from .simulation import LogicalQPU, MemoryNoise, MemoryResult, simulate_memory, effective_logical_channel, wilson_interval
from .recovery import knill_laflamme, transpose_recovery
from .fusion import bell_analyzer_circuit, bell_measurement, bell_instruments
from .decoding import MaximumLikelihoodDecoder, ExactMemoryResult, exact_pauli_memory

from .scalable import SparseBudget, SparseKet, TrajectoryResult, run_sparse, sample_trajectories
from .hardware import FeedForward, TeleportationInstrument, teleportation_instrument
from .reproduction import evidence_record, compare_experiment
