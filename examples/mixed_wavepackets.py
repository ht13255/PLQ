"""Independent mixed spectral states retain three-photon density invariants."""
import numpy as np
from plq import Circuit, mixed_wavepacket_input

rho = np.diag([.7, .3])
tritter = np.exp(-2j*np.pi*np.outer(np.arange(3), np.arange(3))/3)/np.sqrt(3)
result = mixed_wavepacket_input(3, [0, 1, 2], [rho, rho, rho]).through(Circuit(3).unitary(tritter))
expected = (2-3*np.trace(rho@rho).real+4*np.trace(rho@rho@rho).real)/9
print("P(1,1,1):", result.spatial_probabilities()[(1, 1, 1)])
print("Independent trace-invariant formula:", expected)
