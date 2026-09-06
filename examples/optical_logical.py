"""Encoded state -> actual optical rail swap and loss -> weighted code recovery."""
import numpy as np
from plq import Circuit,DualRail,LogicalQPU,PAULI,repetition_code

code=repetition_code(3)
psi=np.array([np.sqrt(.3),1j*np.sqrt(.7)])
register=LogicalQPU(code).prepare(psi)
rails=DualRail(3)
optical=Circuit(6).unitary(PAULI["X"],modes=[2,3])  # a physical single-qubit bit flip
for mode in range(6):
    optical.loss(mode,.98)
result=optical.run(rails.encode(register.rho))
survival=rails.decode(result.state)  # ideal one-photon-per-rail-pair filter; weight retained
register.prepare_physical(survival.rho,subnormalized=True).correct()
logical=register.logical_state()
print("Full optical trace:",result.state.trace)
print("Computational survival (analytic 0.98**3):",logical.probability)
print("Conditional fidelity:",float(np.vdot(psi,logical.conditional()@psi).real))
print("The complement is failure in this example; it is not claimed to be corrected photon loss.")
