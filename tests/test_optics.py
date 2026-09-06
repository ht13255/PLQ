from itertools import permutations
from math import comb, factorial
import numpy as np
import pytest
from plq import *
from plq.optics import lift_unitary, loss_operators, phase_kernel
from plq.numerics import density


def test_hong_ou_mandel_and_vacuum():
    basis = FockBasis(2,2)
    circuit = Circuit(2).bs(0,1)
    out = circuit.run(FockState.ket(basis,(1,1)))
    assert out.probabilities()[(1,1)] == pytest.approx(0, abs=1e-14)
    assert out.probabilities()[(2,0)] == pytest.approx(.5)
    assert out.probabilities()[(0,2)] == pytest.approx(.5)
    assert out.state.trace == pytest.approx(1)
    assert circuit.run(FockState.ket(basis,(0,0))).probabilities()[(0,0)] == 1


@pytest.mark.parametrize("eta",[0,.17,.8,1])
def test_loss_binomial_and_coherences(eta):
    basis = FockBasis(1,3)
    state = Circuit(1).loss(0,eta).run(FockState.ket(basis,(3,))).state
    for k in range(4):
        assert state.probabilities()[(k,)] == pytest.approx(comb(3,k)*eta**k*(1-eta)**(3-k))
    k = loss_operators(basis,0,eta)
    assert np.allclose(sum(a.conj().T@a for a in k),np.eye(4),rtol=0,atol=1e-12)
    superposition = FockState.amplitudes(basis,{(0,):1/np.sqrt(2),(1,):1/np.sqrt(2)})
    output = Circuit(1).loss(0,eta).run(superposition).state.rho
    assert output[0,1] == pytest.approx(np.sqrt(eta)/2)


def test_mach_zehnder_convention():
    basis=FockBasis(2,1)
    phi=.72
    c=Circuit(2).bs(0,1).phase(0,phi).bs(0,1)
    p=c.run(FockState.ket(basis,(1,0))).probabilities()
    assert p[(1,0)] == pytest.approx(np.sin(phi/2)**2)
    assert p[(0,1)] == pytest.approx(np.cos(phi/2)**2)


def test_correlated_phase_noise_exact_gaussian_average():
    basis=FockBasis(2,1)
    state=FockState.amplitudes(basis,{(1,0):1/np.sqrt(2),(0,1):1/np.sqrt(2)})
    covariance=np.array([[.4,.13],[.13,.2]])
    mu=np.array([.2,-.1])
    out=Circuit(2).phase_noise(covariance,mu).run(state).state
    i,j=basis.index[(1,0)],basis.index[(0,1)]
    assert out.rho[i,j] == pytest.approx(.5*np.exp(-.5*(.4+.2-2*.13)+.3j))
    common=Circuit(2).phase_noise(np.ones((2,2))*.5).run(state).state
    assert np.allclose(common.rho,state.rho)
    for bad in (np.array([[1,2],[2,1]]),np.array([[1,0],[1,1]])):
        with pytest.raises(ValueError): Circuit(2).phase_noise(bad)


def test_unequal_rail_loss_is_state_dependent_filter():
    rails=DualRail(1)
    c=Circuit(2).loss(0,.9).loss(1,.4)
    zero=c.run(rails.encode([1,0])).state
    one=c.run(rails.encode([0,1])).state
    assert rails.decode(zero).probability == pytest.approx(.9)
    assert rails.decode(one).probability == pytest.approx(.4)
    branch=rails.decode(c.run(rails.encode(np.ones(2)/np.sqrt(2))).state)
    assert branch.probability == pytest.approx(.65)
    assert branch.rho[0,1] == pytest.approx(np.sqrt(.9*.4)/2)
    assert branch.conditional()[0,0] == pytest.approx(.9/1.3)


def test_loss_map_composition_matches_density_and_flag():
    rails=DualRail(1)
    circuit=Circuit(2).bs(0,1,transmission=.38,phase=.3).loss(0,.73).phase_noise([[.02,0],[0,.1]])
    state=rails.encode([np.sqrt(.3),1j*np.sqrt(.7)])
    assert np.allclose(circuit.channel(rails.basis).apply(state.rho).rho,circuit.run(state).state.rho,atol=1e-12,rtol=0)
    effective=rails.effective_channel(circuit)
    flagged=effective.flagged()
    initial=np.array([np.sqrt(.3),1j*np.sqrt(.7),0,0])
    out=flagged.apply(initial).rho
    branch=rails.decode(circuit.run(state).state)
    assert np.allclose(out[:2,:2],branch.rho)
    assert out[2,2].real == pytest.approx(1-branch.probability)
    assert np.trace(out).real == pytest.approx(1)


def test_source_tail_is_reported_not_renormalized():
    source=independent_sources(FockBasis(2,1),[[.1,.8,.1],[.2,.8]])
    assert source.state.trace == pytest.approx(.26)
    assert source.omitted_probability == pytest.approx(.74)


@pytest.mark.parametrize("overlap",[0,.3,.7,1,.3+.6j])
def test_partial_distinguishability_hom(overlap):
    photons=wavepacket_input(2,[0,1],[[1,overlap],[np.conj(overlap),1]])
    p=photons.through(Circuit(2).bs(0,1)).spatial_probabilities()
    assert p.get((1,1),0) == pytest.approx((1-abs(overlap)**2)/2,abs=1e-12)
    assert sum(p.values()) == pytest.approx(1)


def test_three_photon_complex_gram_against_permutation_formula():
    rng=np.random.default_rng(75)
    internal=rng.normal(size=(2,3))+1j*rng.normal(size=(2,3))
    internal/=np.linalg.norm(internal,axis=0)
    gram=internal.conj().T@internal
    u=np.exp(2j*np.pi*np.outer(np.arange(3),np.arange(3))/3)/np.sqrt(3)
    distribution=wavepacket_input(3,[0,1,2],gram).through(Circuit(3).unitary(u)).spatial_probabilities()
    for occupation,p in distribution.items():
        if sum(occupation)!=3: continue
        outputs=[i for i,n in enumerate(occupation) for _ in range(n)]
        total=0j
        for a in permutations(range(3)):
            for b in permutations(range(3)):
                total+=np.prod([u[outputs[k],a[k]]*u[outputs[k],b[k]].conj()*gram[b[k],a[k]] for k in range(3)])
        total/=np.prod([factorial(n) for n in occupation])
        assert p == pytest.approx(total.real,abs=1e-11)


def test_gaussian_overlap_delay_and_loss():
    sigma=30e-12
    dt=2*sigma
    g=gaussian_gram([0,dt],[0,0],sigma)
    assert abs(g[0,1])**2 == pytest.approx(np.exp(-1))
    photons=wavepacket_input(2,[0,1],g)
    c=Circuit(2).bs(0,1).loss(0,.8).loss(1,.8)
    p=photons.through(c).spatial_probabilities()
    assert sum(v for s,v in p.items() if sum(s)==2) == pytest.approx(.8**2)
    assert p[(1,1)] == pytest.approx(.8**2*(1-np.exp(-1))/2)


def test_validation_and_resource_limits():
    with pytest.raises(ValueError): density([[1,.5],[0,0]])
    with pytest.raises(ValueError): density([[1.1,0],[0,-.1]])
    with pytest.raises(ValueError): Circuit(2).bs(0,0)
    with pytest.raises(ValueError): Circuit(1).loss(0,-.1)
    with pytest.raises(ValueError): Circuit(1).unitary([[.9]])
    with pytest.raises(ValueError): FockState.ket(FockBasis(1,1),(2,))
    with pytest.raises(ResourceLimitError): FockBasis(20,10)
    with pytest.raises(ValueError): wavepacket_input(2,[0,1],[[1,1.1],[1.1,1]])
    with pytest.raises(ValueError): Branch(np.zeros((2,2))).conditional()


def test_independent_high_precision_permanent_reference():
    pytest.importorskip("mpmath")
    from plq.reference import fock_amplitude
    rng=np.random.default_rng(3)
    u,_=np.linalg.qr(rng.normal(size=(3,3))+1j*rng.normal(size=(3,3)))
    basis=FockBasis(3,3)
    lifted=lift_unitary(u,basis)
    assert np.allclose(lifted.conj().T@lifted,np.eye(basis.dimension),rtol=0,atol=2e-14)
    for i,target in enumerate(basis.states):
        for j,source in enumerate(basis.states):
            assert lifted[i,j] == pytest.approx(complex(fock_amplitude(u,source,target,digits=70)),abs=2e-14)
