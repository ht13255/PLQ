"""Use a measured optical channel inside a differentiable PennyLane algorithm."""
import pennylane as qml
from pennylane import numpy as np
from plq import Circuit,DualRail
from plq.adapters.pennylane import to_operation

channel=DualRail(1).effective_channel(Circuit(2).loss(0,.7).loss(1,.9)).flagged()
device=qml.device("default.mixed",wires=["flag","data"])

@qml.qnode(device, diff_method="parameter-shift")
def algorithm(theta):
    qml.RY(theta,wires="data")
    to_operation(channel,["flag","data"])
    return qml.expval(qml.Projector([0],wires="flag"))

theta=np.array(.6,requires_grad=True)
print("Survival probability:",algorithm(theta))
gradient=qml.grad(algorithm)(theta)
expected=.1*np.sin(theta)
assert np.isfinite(gradient) and np.allclose(gradient,expected,atol=1e-10,rtol=0)
print("Gradient of survival with respect to the algorithm angle:",gradient)
print("The optical channel is fixed; PLQ's NumPy optical simulation itself is not autodifferentiable.")
