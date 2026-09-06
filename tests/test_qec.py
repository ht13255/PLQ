from itertools import product, combinations
import numpy as np
import pytest
from plq import *
from plq.qec import parse_pauli, pauli_matrix
from plq.optics import loss_operators


@pytest.mark.parametrize("factory",[five_qubit_code,steane_code,shor_code])
def test_every_single_pauli_error_corrected(factory):
    code=factory()
    decoder=MinimumWeightDecoder(code)
    for q in range(code.n):
        for error in "XYZ":
            word="I"*q+error+"I"*(code.n-q-1)
            _,_,x,z=parse_pauli(word)
            _,_,cx,cz=parse_pauli(decoder.decode(code.syndrome(word)))
            assert code.in_stabilizer(x^cx,z^cz)


def test_dense_coherent_error_recovery_unknown_logical_state():
    code=five_qubit_code()
    psi=np.array([np.sqrt(.37),np.exp(.43j)*np.sqrt(.63)])
    theta=.23
    rotation=np.cos(theta)*np.eye(2)-1j*np.sin(theta)*PAULI["X"]
    qpu=LogicalQPU(code).prepare(psi).physical_gate(rotation,[2]).correct()
    assert qpu.logical_state().probability == pytest.approx(1)
    assert np.allclose(qpu.logical_state().rho,np.outer(psi,psi.conj()),rtol=0,atol=2e-12)


def test_knill_laflamme_for_five_qubit_code():
    code=five_qubit_code()
    errors=[np.eye(32)]+[pauli_matrix("I"*q+p+"I"*(4-q)) for q in range(5) for p in "XYZ"]
    assert knill_laflamme(code,errors)["max_absolute_residual"] < 1e-12


def test_erasure_decoder_corrects_every_pair_in_five_qubit_code():
    code=five_qubit_code()
    decoder=ErasureDecoder(code)
    for erased in combinations(range(5),2):
        for errors in product("IXYZ",repeat=2):
            word=["I"]*5
            for q,e in zip(erased,errors): word[q]=e
            _,_,x,z=parse_pauli("".join(word))
            correction=decoder.decode(code.syndrome_masks(x,z),erasures=erased)
            _,_,cx,cz=parse_pauli(correction)
            assert code.in_stabilizer(x^cx,z^cz)


def test_repetition_exact_bitflip_rate_and_monte_carlo():
    p=.08
    code=repetition_code()
    qpu=LogicalQPU(code)
    for q in range(3): qpu.local_channel(pauli_channel(px=p),[q])
    qpu.correct()
    expected=3*p*p-2*p**3
    assert qpu.logical_state().rho[1,1].real == pytest.approx(expected,abs=1e-12)
    result=simulate_memory(code,MemoryNoise(px=p),shots=30000,seed=77)
    assert result.logical_error_rate == pytest.approx(expected,abs=5*np.sqrt(expected*(1-expected)/30000))
    assert result.wilson_95[0] <= result.logical_error_rate <= result.wilson_95[1]


def test_multiround_perfect_readout_and_seed():
    code=repetition_code()
    options=dict(shots=100,rounds=3,seed=12)
    a=simulate_memory(code,MemoryNoise(px=.03,syndrome_flip=.02),**options)
    b=simulate_memory(code,MemoryNoise(px=.03,syndrome_flip=.02),**options)
    assert a.to_dict() == b.to_dict()
    assert simulate_memory(code,shots=50,rounds=5).failures == 0
    assert wilson_interval(0,100)[1] > 0


def test_custom_decoder_gets_observations_and_flags():
    code=five_qubit_code()
    class Tracking:
        def __init__(self):
            self.delegate=ErasureDecoder(code)
            self.calls=[]
        def decode(self,syndrome,*,erasures=(),history=()):
            self.calls.append((erasures,len(history)))
            return self.delegate.decode(syndrome,erasures=erasures)
    decoder=Tracking()
    result=simulate_memory(code,MemoryNoise(erasure=[1,0,0,0,0]),shots=3,rounds=2,decoder=decoder)
    assert result.failures == 0
    assert ((0,),1) in decoder.calls
    assert ((),2) in decoder.calls


def test_signed_stabilizer_and_multilogical_spaces():
    signed=StabilizerCode(["-ZZ"],logical_x=["XX"],logical_z=["ZI"])
    v=signed.codespace.isometry
    assert np.allclose(pauli_matrix("-ZZ")@v,v)
    assert np.allclose(v.conj().T@pauli_matrix("XX")@v,PAULI["X"])
    multi=StabilizerCode(["XXXX","ZZZZ"])
    assert multi.codespace.isometry.shape == (16,4)
    bell=StabilizerCode(["XX","ZZ"])
    assert bell.codespace.isometry.shape == (4,1)


def test_custom_css_and_serialization(tmp_path):
    code=css_code(np.zeros((0,3),int),[[1,1,0],[0,1,1]],logical_x=["XXX"],logical_z=["ZII"])
    assert code.syndrome("IXI") == (1,1)
    path=tmp_path/"code.json"
    code.save(path)
    assert StabilizerCode.load(path).to_dict()==code.to_dict()
    path=tmp_path/"code.npz"
    code.codespace.save(path)
    assert np.allclose(CodeSpace.load(path).isometry,code.codespace.isometry)
    register_code("my_repetition_for_test",lambda:code)
    assert get_code("my_repetition_for_test") is code


def test_arbitrary_bosonic_code_and_transpose_recovery():
    # Finite binomial code in the ordered Fock basis |0>,...,|4>.
    basis=FockBasis(1,4)
    v=np.zeros((5,2),complex)
    v[0,0]=v[4,0]=1/np.sqrt(2)
    v[2,1]=1
    code=CodeSpace(v,name="finite binomial")
    annihilation=np.diag(np.sqrt(np.arange(1,5)),1)
    assert knill_laflamme(code,[np.eye(5),annihilation])["satisfies_tolerance"]
    noise=KrausChannel(loss_operators(basis,0,.98))
    recovery=transpose_recovery(code,noise)
    psi=np.array([1,1j])/np.sqrt(2)
    qpu=LogicalQPU(code).prepare(psi).channel(noise).correct(recovery=recovery)
    branch=qpu.logical_state()
    fidelity=np.vdot(psi,branch.rho@psi).real
    assert branch.probability == pytest.approx(1,abs=1e-10)
    assert .99 < fidelity < 1  # loss at finite eta is not claimed perfectly correctable


def test_no_silent_invalid_code_or_decoder():
    with pytest.raises(ValueError): StabilizerCode(["XI","ZI"])
    with pytest.raises(ValueError): StabilizerCode(["ZZ","-ZZ"])
    with pytest.raises(ValueError): StabilizerCode(["II"])
    with pytest.raises(ValueError): css_code([[1,0]],[[1,0]])
    with pytest.raises(ValueError): CodeSpace([[1,1],[0,0]])
    with pytest.raises(ValueError): KrausChannel([np.eye(2)*1.01])
    with pytest.raises(ValueError): KrausChannel([np.eye(2)*1.01],trace_preserving=False)
    with pytest.raises(ValueError): simulate_memory(repetition_code(),MemoryNoise(px=.7,pz=.4),shots=1)
    with pytest.raises(DecodeFailure): MinimumWeightDecoder(repetition_code()).decode((0,0),erasures=(0,))
