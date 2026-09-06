import numpy as np
import pytest
from plq import *


def test_perceval_unitary_and_fock_probabilities():
    pcvl=pytest.importorskip("perceval")
    from plq.adapters.perceval import from_perceval,to_perceval,reference_probabilities
    optical=pcvl.Circuit(3).add((0,1),pcvl.BS.H()).add(1,pcvl.PS(.37)).add((1,2),pcvl.BS.Ry(theta=.63))
    local=from_perceval(optical)
    assert np.allclose(np.array(to_perceval(local).compute_unitary()),np.array(optical.compute_unitary()))
    basis=FockBasis(3,3)
    for source in [(1,1,0),(2,0,1),(0,3,0)]:
        probabilities=local.run(FockState.ket(basis,source)).probabilities()
        reference=reference_probabilities(local,source)
        assert sum(reference.values()) == pytest.approx(1,abs=1e-11)
        for occupation,p in probabilities.items():
            assert p == pytest.approx(reference.get(occupation,0),abs=1e-11)


def test_perceval_refuses_processor_and_noisy_export():
    pcvl=pytest.importorskip("perceval")
    from plq.adapters.perceval import from_perceval,to_perceval
    with pytest.raises(TypeError): from_perceval(pcvl.Processor("SLOS",2))
    with pytest.raises(ValueError): to_perceval(Circuit(2).loss(0,.9))


def test_pennylane_channels_and_wire_order():
    qml=pytest.importorskip("pennylane")
    from plq.adapters.pennylane import from_operations
    wire_order=["second","first"]
    def algorithm():
        qml.RY(.61,wires="first")
        qml.CNOT(wires=["first","second"])
        qml.AmplitudeDamping(.13,wires="second")
        qml.PhaseFlip(.2,wires="first")
    tape=qml.tape.make_qscript(algorithm)()
    channel=from_operations(tape.operations,wire_order=wire_order)
    device=qml.device("default.mixed",wires=wire_order)
    @qml.qnode(device)
    def reference():
        algorithm()
        return qml.state()
    assert np.allclose(channel.apply([1,0,0,0]).rho,reference(),atol=2e-12,rtol=0)
    with pytest.raises(ValueError): from_operations([qml.StatePrep([1,0],wires=0)],wire_order=[0])


def test_pennylane_flagged_optical_channel_retains_failures():
    qml=pytest.importorskip("pennylane")
    from plq.adapters.pennylane import to_operation
    channel=DualRail(1).effective_channel(Circuit(2).loss(0,.6).loss(1,.8))
    with pytest.raises(ValueError): to_operation(channel,[0])
    device=qml.device("default.mixed",wires=["flag","data"])
    @qml.qnode(device)
    def run():
        qml.Hadamard("data")
        to_operation(channel.flagged(),["flag","data"])
        return qml.probs(wires=["flag","data"])
    assert np.allclose(run(),[.3,.4,.3,0],atol=1e-12,rtol=0)


def test_pennylane_logical_algorithm_runs_in_code():
    qml=pytest.importorskip("pennylane")
    from plq.adapters.pennylane import logical_unitary
    def algorithm(angle):
        qml.Hadamard(0)
        qml.RZ(angle,0)
    u=logical_unitary(algorithm,.41,wire_order=[0])
    qpu=LogicalQPU(five_qubit_code()).logical_gate(u).physical_gate(PAULI["Y"],[1]).correct()
    assert np.allclose(qpu.logical_state().rho,np.outer(u[:,0],u[:,0].conj()),atol=2e-12,rtol=0)


def test_pennylane_gradient_matches_analytic_derivative():
    qml=pytest.importorskip("pennylane")
    from pennylane import numpy as anp
    from plq.adapters.pennylane import to_operation
    fixed=DualRail(1).effective_channel(Circuit(2).loss(0,.7).loss(1,.9)).flagged()
    @qml.qnode(qml.device("default.mixed",wires=[0,1]),diff_method="parameter-shift")
    def probability(theta):
        qml.RY(theta,1)
        to_operation(fixed,[0,1])
        return qml.expval(qml.Projector([0],wires=0))
    theta=anp.array(.6,requires_grad=True)
    gradient=qml.grad(probability)(theta)
    assert np.isfinite(gradient)
    assert gradient == pytest.approx(.1*np.sin(.6),abs=1e-11)


@pytest.mark.parametrize("p",[0,.008])
def test_stim_surface_code_and_matching(p):
    stim=pytest.importorskip("stim")
    pytest.importorskip("pymatching")
    from plq.adapters.stim import simulate_stim
    circuit=stim.Circuit.generated("surface_code:rotated_memory_z",distance=3,rounds=3,
                                    after_clifford_depolarization=p,before_measure_flip_probability=p)
    result=simulate_stim(circuit,shots=2000,seed=123)
    assert result.detectors==circuit.num_detectors
    assert result.shots==2000
    if p==0: assert result.failures==0
    else: assert 0 < result.failures < 400
    with pytest.raises(ValueError): simulate_stim("H 0\nM 0",shots=1)
