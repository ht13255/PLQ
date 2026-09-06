import perceval as pcvl
from plq import FockBasis,FockState
from plq.adapters.perceval import from_perceval,reference_probabilities

design=pcvl.Circuit(2).add(0,pcvl.BS.H()).add(1,pcvl.PS(.4))
local=from_perceval(design)
basis=FockBasis(2,2)
print("PLQ:",{s:p for s,p in local.run(FockState.ket(basis,(1,1))).probabilities().items() if p>1e-12})
print("Perceval SLOS:",reference_probabilities(local,(1,1)))
# Explicitly add physical noise after interchange. Processor noise is not guessed.
local.loss(0,.9)
print("Noisy trace:",local.run(FockState.ket(basis,(1,1))).state.trace)
