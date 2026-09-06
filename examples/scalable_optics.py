"""A connected passive mesh beyond the default density budget, plus lossy shots."""
from plq import Circuit, SparseBudget

circuit = Circuit(16)
for layer in range(3):
    for mode in range(layer % 2, 15, 2):
        circuit.bs(mode, mode+1, transmission=.43, phase=.17*layer)
occupation = [int(mode in (1, 5, 9, 13)) for mode in range(16)]
result = circuit.run_sparse(occupation, budget=SparseBudget(max_terms=20_000))
print({'support_terms': len(result.amplitudes), 'norm_squared': result.norm})
# Explicitly switch to Monte Carlo for loss. The result is not an exact density.
small = Circuit(2).bs(0, 1).loss(0, .85).loss(1, .85)
print(small.sample([1, 1], shots=2000, seed=7).to_dict())
