import numpy as np
import pytest
from plq import (teleportation_instrument, FeedForward, Detector, FockState, FockBasis,
                 LogicalQPU, repetition_code, Circuit)


@pytest.fixture(scope='module')
def ideal():
    return teleportation_instrument()


@pytest.mark.parametrize('psi', [[1, 0], [0, 1], [1, 1], [1, -1], [1, 1j], [1, -1j]])
def test_teleportation_is_identity_for_six_states_and_each_herald(ideal, psi):
    psi = np.asarray(psi)/np.linalg.norm(psi)
    result = ideal.apply(psi)
    expected = np.outer(psi, psi.conj())
    assert result['accepted_probability'] == pytest.approx(.5)
    assert np.allclose(result['computational_branch'].rho, .5*expected, atol=2e-14)
    v = np.zeros((ideal.output_basis.dimension, 2))
    v[ideal.output_basis.index[(1, 0)], 0] = v[ideal.output_basis.index[(0, 1)], 1] = 1
    for branch in result['accepted_events'].values():
        assert branch.probability == pytest.approx(.125)
        assert np.allclose(v.T@branch.rho@v, .125*expected, atol=2e-14)
    assert ideal.flagged_channel().completeness_residual < 1e-13


def test_output_loss_is_not_falsely_exposed_as_a_heralded_erasure():
    h = teleportation_instrument(transmissions=[1, 1, 1, 1, .6, .6])
    r = h.apply([1, 0])
    assert r['accepted_probability'] == pytest.approx(.5)
    assert r['computational_branch'].probability == pytest.approx(.3)
    assert r['accepted_leakage_probability'] == pytest.approx(.2)
    assert r['rejected_probability'] == pytest.approx(.5)
    assert h.flagged_channel().apply([1, 0]).probability == pytest.approx(1)


def test_measurement_loss_and_detector_efficiency_match_photon_budget():
    h = teleportation_instrument(transmissions=[.8]*6, detectors=[Detector(efficiency=.7)]*4)
    r = h.apply([1, 0])
    assert r['accepted_probability'] == pytest.approx(.5*.8**2*.7**2)
    assert r['computational_branch'].probability == pytest.approx(.5*.8**3*.7**2)


def test_dark_false_heralds_preserve_output_vacuum():
    resource = FockState.ket(FockBasis(4, 0), [0]*4)
    h = teleportation_instrument(resource=resource, detectors=[Detector(threshold=True, dark_rate_hz=1e8)]*4)
    r = h.apply([1, 0])
    assert r['accepted_probability'] > 0
    assert r['computational_branch'].probability == pytest.approx(0, abs=1e-14)
    assert r['accepted_leakage_probability'] == pytest.approx(r['accepted_probability'])
    assert h.flagged_channel().completeness_residual < 1e-13


def test_deadline_and_buffer_attenuation_are_separate():
    late = teleportation_instrument(schedule=FeedForward(processing_seconds=2e-9, buffer_seconds=1e-9))
    r = late.apply([1, 0])
    assert r['accepted_probability'] == 0
    assert r['late_probability'] == pytest.approx(.5)
    assert late.flagged_channel().completeness_residual < 1e-13
    schedule = FeedForward(processing_seconds=1e-9, buffer_seconds=2e-9, buffer_loss_db_per_second=1e9,
                           switch_transmission=.9)
    h = teleportation_instrument(schedule=schedule)
    r = h.apply([1, 0])
    assert r['computational_branch'].probability == pytest.approx(.5*.9*10**(-.2))


def test_calibrated_analyzer_and_no_pauli_twirl():
    transfer = Circuit(4).bs(0, 2).bs(1, 3).single_particle_unitary()
    h = teleportation_instrument(analyzer_transfer=transfer, beamsplitter_transmission=.5, output_phase=.2)
    # Averaging the two corrected Bell labels can itself produce dephasing;
    # individual conditional maps retain their opposite coherent phases.
    r = h.apply(np.array([1, 1])/np.sqrt(2))
    assert r['accepted_probability'] == pytest.approx(.5)
    off_diagonal = [branch.rho[h.output_basis.index[(1, 0)], h.output_basis.index[(0, 1)]].imag
                    for branch in r['accepted_events'].values()]
    assert max(abs(v) for v in off_diagonal) > .001
    with pytest.raises(ValueError): teleportation_instrument(analyzer_transfer=1.1*np.eye(4))


def test_optical_channel_composes_with_small_code_without_renormalization(ideal):
    psi = np.array([1, 1j])/np.sqrt(2)
    qpu = LogicalQPU(repetition_code(3)).prepare(psi)
    for wire in range(3):
        qpu.local_channel(ideal.computational_channel(), [wire])
    result = qpu.correct().logical_state()
    assert result.probability == pytest.approx(.5**3)
    assert np.allclose(result.conditional(), np.outer(psi, psi.conj()), atol=2e-14)


def test_resource_and_schedule_validation():
    with pytest.raises(ValueError): FeedForward(processing_seconds=-1)
    with pytest.raises(ValueError): teleportation_instrument(detectors=[Detector(saturation=1)]*4)
    with pytest.raises(ValueError): teleportation_instrument(transmissions=[.9])


def test_state_dependent_acceptance_remains_a_linear_map():
    h = teleportation_instrument(transmissions=[.8, .4, 1, 1, 1, 1])
    assert h.apply([1, 0])['accepted_probability'] == pytest.approx(.4)
    assert h.apply([0, 1])['accepted_probability'] == pytest.approx(.2)
    assert h.apply(np.eye(2)/2)['accepted_probability'] == pytest.approx(.3)
    assert h.flagged_channel().completeness_residual < 1e-13
