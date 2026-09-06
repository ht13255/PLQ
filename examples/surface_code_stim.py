"""Circuit-level surface-code example using an explicit Stim circuit and PyMatching."""
import json
import stim
from plq.adapters.stim import simulate_stim

circuit=stim.Circuit.generated("surface_code:rotated_memory_z",distance=5,rounds=5,
    after_clifford_depolarization=.003,before_measure_flip_probability=.003)
print(json.dumps(simulate_stim(circuit,shots=10000,seed=2026).to_dict(),indent=2))
# Custom circuits: circuit=stim.Circuit.from_file("my_fusion_memory.stim")
# Your circuit must explicitly contain the intended noise, detectors and observables.
