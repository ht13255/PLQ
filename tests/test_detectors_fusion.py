import numpy as np
import pytest
from plq import *


def test_dark_counts_efficiency_saturation_normalization():
    d=Detector(efficiency=.7,dark_rate_hz=2e8,gate_seconds=1e-9,saturation=3)
    for n in (0,1,2,5,12):
        response=d.response(n)
        assert response.sum() == pytest.approx(1)
        assert response[0] == pytest.approx(.3**n*np.exp(-.2))
    threshold=Detector(efficiency=.7,dark_rate_hz=2e8,gate_seconds=1e-9,threshold=True)
    assert threshold.response(2)[1] == pytest.approx(1-.3**2*np.exp(-.2))


def test_timing_units():
    assert Detector(gate_seconds=1e-9,arrival_offset_seconds=2e-9).effective_efficiency == 0
    assert Detector(gate_seconds=0).effective_efficiency == 0
    d=Detector(gate_seconds=2e-9,timing_sigma_seconds=1e-9)
    assert d.effective_efficiency == pytest.approx(.682689492137)
    with pytest.raises(ValueError): Detector(efficiency=1.1)


def test_herald_traces_measured_modes_and_keeps_weight():
    basis=FockBasis(2,2)
    state=FockState.amplitudes(basis,{(1,0):1/np.sqrt(2),(0,1):1/np.sqrt(2)})
    result=herald(state,[0],[1],[Detector(efficiency=.8,saturation=2)])
    assert result.probability == pytest.approx(.4)
    assert result.remaining_state.trace == pytest.approx(.4)
    assert result.conditional_state().probabilities()[(0,)] == pytest.approx(1)
    no_click=herald(state,[0],[0],[Detector(efficiency=.8,saturation=2)])
    assert no_click.probability == pytest.approx(.6)
    assert no_click.remaining_state.rho[0,1] == 0  # destructive trace removes inter-number coherence
    probabilities=detection_probabilities(state,[0],[Detector(efficiency=.8,saturation=2)])
    assert sum(probabilities.values()) == pytest.approx(1)
    assert probabilities[(0,)] == pytest.approx(no_click.probability)


def test_bell_analyzer_50_percent_and_complete_instrument():
    bell=[np.array(v)/np.sqrt(2) for v in ([1,0,0,1],[1,0,0,-1],[0,1,1,0],[0,1,-1,0])]
    results=[bell_measurement(v) for v in bell]
    assert sum(r["psi_plus"]+r["psi_minus"] for r in results)/4 == pytest.approx(.5)
    assert results[2]["psi_plus"] == pytest.approx(1)
    assert results[3]["psi_minus"] == pytest.approx(1)
    instruments=bell_instruments()
    total=sum(k.conj().T@k for channel in instruments.values() for k in channel.operators)
    assert np.allclose(total,np.eye(4),atol=1e-12,rtol=0)
    for v,result in zip(bell,results):
        for name,channel in instruments.items():
            assert channel.apply(v).probability == pytest.approx(result[name])


def test_loss_and_false_herald_are_not_idealized_away():
    psi=np.array([0,1,1,0])/np.sqrt(2)
    assert bell_measurement(psi,transmission=.8)["psi_plus"] == pytest.approx(.64)
    noisy=(Detector(efficiency=.8,dark_rate_hz=1e8,gate_seconds=1e-9,saturation=3),)*4
    result=bell_measurement(np.array([1,0,0,1])/np.sqrt(2),detectors=noisy)
    assert result["psi_plus"]+result["psi_minus"] > 0  # false heralds
    assert sum(result.values()) == pytest.approx(1)


def test_wavepacket_grouped_detection_matches_spatial_model():
    photons=wavepacket_input(2,[0,1],[[1,.6],[.6,1]]).through(Circuit(2).bs(0,1))
    detectors=(Detector(threshold=True),)*2
    outcomes=photons.detection_probabilities(detectors)
    assert outcomes[(1,1)] == pytest.approx((1-.6**2)/2)
    assert sum(outcomes.values()) == pytest.approx(1)
