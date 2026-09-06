"""Compare two decoders under the SAME fully specified one-round Pauli model."""
from plq import five_qubit_code, MemoryNoise, MinimumWeightDecoder, exact_pauli_memory

code = five_qubit_code()
noise = MemoryNoise(px=.001, py=.001, pz=.15)
for decoder in (MinimumWeightDecoder(code), None):
    result = exact_pauli_memory(code, noise, decoder=decoder)
    print(f"{result.decoder}: logical block error = {result.logical_error_rate:.12f}, "
          f"patterns = {result.patterns}, total probability = {result.total_probability:.12f}")

# The default sums stabilizer coset probabilities before choosing a correction.
# This is not a threshold estimate or a model of physical syndrome extraction.
