"""Arbitrary finite bosonic code, physical loss, and approximate transpose recovery."""
import numpy as np
from plq import CodeSpace,FockBasis,KrausChannel,LogicalQPU,knill_laflamme,transpose_recovery
from plq.optics import loss_operators

v=np.zeros((5,2),complex)
v[0,0]=v[4,0]=1/np.sqrt(2)
v[2,1]=1
code=CodeSpace(v,name="binomial |0L>=(|0>+|4>)/sqrt(2), |1L>=|2>")
a=np.diag(np.sqrt(np.arange(1,5)),1)
print("{I,a} Knill-Laflamme residual:",knill_laflamme(code,[np.eye(5),a])["max_absolute_residual"])
noise=KrausChannel(loss_operators(FockBasis(1,4),0,.98))
recovery=transpose_recovery(code,noise)
psi=np.array([1,1j])/np.sqrt(2)
result=LogicalQPU(code).prepare(psi).channel(noise).correct(recovery=recovery).logical_state()
print("Finite-loss corrected fidelity:",float(np.vdot(psi,result.rho@psi).real))
