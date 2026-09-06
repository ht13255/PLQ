import json
import numpy as np
import pytest
from plq import Circuit, SparseKet, SparseBudget, ResourceLimitError, Detector
from plq.cli import optical_scenario, main
from plq.reference import fock_amplitude


@pytest.mark.parametrize('occupation', [[1, 1, 0], [2, 1, 0], [0, 0, 0]])
def test_sparse_matches_dense_coherences_and_permanents(occupation):
    rng = np.random.default_rng(18)
    u, _ = np.linalg.qr(rng.normal(size=(3, 3))+1j*rng.normal(size=(3, 3)))
    circuit = Circuit(3).unitary(u)
    result = circuit.run_sparse(occupation)
    assert np.allclose(result.to_fock().rho, circuit.run(occupation).state.rho, atol=2e-14, rtol=0)
    for target, amp in result.amplitudes.items():
        assert amp == pytest.approx(complex(fock_amplitude(u, occupation, target)), abs=2e-14)


def test_number_superposition_and_local_mode_order():
    state = SparseKet(3, {(0, 0, 0): np.sqrt(.2), (2, 0, 0): np.sqrt(.3), (0, 1, 1): 1j*np.sqrt(.5)})
    circuit = Circuit(3).bs(2, 0, transmission=.31, phase=.87).phase(1, -.4).bs(0, 1)
    assert np.allclose(circuit.run_sparse(state).to_fock().rho, circuit.run(state.to_fock()).state.rho, atol=2e-14)


def test_large_sparse_support_without_fock_allocation(monkeypatch):
    import plq.optics
    def forbidden(*args, **kwargs):
        raise AssertionError('Dense Fock allocation attempted')
    monkeypatch.setattr(plq.optics, 'FockBasis', forbidden)
    circuit = Circuit(32)
    occupation = [0]*32
    for mode in range(0, 32, 4):
        occupation[mode] = 1
        circuit.bs(mode, mode+1)
    result = circuit.run_sparse(occupation)
    assert len(result.amplitudes) == 256
    assert result.norm == pytest.approx(1)
    assert all(p == pytest.approx(1/256) for p in result.probabilities().values())


def test_budget_and_unsupported_operations_fail_explicitly():
    with pytest.raises(ResourceLimitError):
        Circuit(2).bs(0, 1).run_sparse([4, 4], budget=SparseBudget(max_terms=2))
    with pytest.raises(ResourceLimitError):
        Circuit(2).bs(0, 1).run_sparse([1, 1], budget=SparseBudget(max_transitions=1))
    with pytest.raises(ResourceLimitError):
        SparseKet.occupation([1000], budget=SparseBudget(max_photons=3))
    with pytest.raises(ValueError):
        Circuit(1).loss(0, .7).run_sparse([1])
    with pytest.raises(ValueError):
        Circuit(1).thermal_loss(0, .7, .1).sample([1], shots=1)


def test_trajectory_loss_phase_and_detector_statistics():
    circuit = Circuit(2).bs(0, 1).phase_noise([[.8, 0], [0, 0]]).bs(0, 1).loss(0, .7).loss(1, .7)
    result = circuit.sample([1, 0], shots=8000, seed=17, detectors=[Detector(threshold=True, efficiency=.6)]*2)
    exact = circuit.run([1, 0]).probabilities()
    for occ, p in exact.items():
        # Broad deterministic five-standard-error bound; checks a physical
        # expectation, not identical implementation internals.
        assert abs(result.counts.get(occ, 0)/result.shots-p) < 5*np.sqrt(p*(1-p)/result.shots)+.003
    detected = sum(v for k, v in result.detector_counts.items() if sum(k))
    assert detected/result.shots == pytest.approx(.7*.6, abs=.03)
    assert sum(result.counts.values()) == result.shots
    assert result.interval((9, 9))[1] > 0
    assert Circuit(1).loss(0, 0).sample([2], shots=10).counts == {(0,): 10}


def test_trajectory_reproducibility_transfer_and_json(tmp_path):
    circuit = Circuit(2).transfer([[.6, .1j], [.2, .5]])
    a = circuit.sample([1, 1], shots=100, seed=44).to_dict()
    b = circuit.sample([1, 1], shots=100, seed=44).to_dict()
    assert a == b
    config = {'modes': 2, 'backend': 'trajectories', 'input': {'occupation': [1, 0]},
              'steps': [{'operation': 'loss', 'mode': 0, 'transmission': .8}], 'shots': 20, 'seed': 2}
    path = tmp_path/'scenario.json'; output = tmp_path/'output.json'
    path.write_text(json.dumps(config))
    main(['optics', str(path), '--output', str(output)])
    assert json.loads(output.read_text())['shots'] == 20
    for extra in ({'herald': {}}, {'precision': {}}, {'max_photons': 2}):
        with pytest.raises(ValueError): optical_scenario({**config, **extra})
    with pytest.raises(ValueError): optical_scenario({**config, 'backend': 'density'})


def test_interleaved_loss_preserves_coherences_before_later_interference():
    circuit = Circuit(2).bs(0, 1, transmission=.31).loss(0, .4).bs(0, 1, phase=.7)
    result = circuit.sample([2, 1], shots=4000, seed=13)
    expected = circuit.run([2, 1]).probabilities()
    for occ, p in expected.items():
        assert abs(result.counts.get(occ, 0)/result.shots-p) < 5*np.sqrt(p*(1-p)/result.shots)+.003
