"""Independent moments, density invariants and vacuum-dilation comparisons."""
from itertools import product
import json
import numpy as np
import pytest
from scipy.linalg import sqrtm
from plq import (Circuit, Detector, FockBasis, FockState, Precision, ResourceLimitError,
                 herald, independent_sources, mixed_wavepacket_input,
                 number_distribution_from_moments, number_moments, source_hom,
                 wavepacket_input, wavepacket_sources)
from plq.cli import main, optical_scenario, source_hom_scenario


@pytest.mark.parametrize("overlap", [1-1e-7, 1-1e-12, 1-1e-14])
def test_positive_gram_eigenvalues_preserve_rare_hom_events(overlap):
    state = wavepacket_input(2, [0, 1], [[1, overlap], [overlap, 1]])
    result = state.through(Circuit(2).bs(0, 1)).spatial_probabilities()
    expected = (1-overlap)*(1+overlap)/2
    assert state.internal_modes == 2
    assert result[(1, 1)] == pytest.approx(expected, rel=2e-12, abs=0)


@pytest.mark.parametrize("mean,g2", [(.154, .0028), (.66, .009), (.65, .024), (1.8, .5), (1e-12, 1.)])
def test_source_moments_recover_published_and_limiting_parameters(mean, g2):
    p = number_distribution_from_moments(mean, g2)
    assert np.all(p >= 0)
    assert sum(p) == pytest.approx(1, abs=2e-15)
    moments = number_moments(p)
    assert moments["mean_photons"] == pytest.approx(mean, rel=1e-14, abs=0)
    assert moments["g2_zero"] == pytest.approx(g2, rel=1e-14, abs=0)
    assert p[2] != pytest.approx(g2)  # g2 is a normalized factorial moment.


@pytest.mark.parametrize("mean,g2", [(0, 0), (-1, .1), (3, .1), (1.8, 0), (.9, 2), (.1, np.inf), (np.nan, 0)])
def test_incompatible_moments_are_rejected(mean, g2):
    with pytest.raises(ValueError):
        number_distribution_from_moments(mean, g2)


def test_mixed_internal_states_match_density_invariants():
    rhos = [np.array([[.7, .1j], [-.1j, .3]]),
            np.array([[.4, .2], [.2, .6]]), np.diag([.8, .2])]
    for a, b in ((0, 1), (1, 2), (0, 2)):
        result = mixed_wavepacket_input(2, [0, 1], [rhos[a], rhos[b]])
        p = result.through(Circuit(2).bs(0, 1)).spatial_probabilities()
        assert p[(1, 1)] == pytest.approx((1-np.trace(rhos[a]@rhos[b]).real)/2, abs=3e-14)
    u = np.exp(2j*np.pi*np.outer(np.arange(3), np.arange(3))/3)/np.sqrt(3)
    result = mixed_wavepacket_input(3, [0, 1, 2], rhos).through(Circuit(3).unitary(u))
    # Menssen et al., supplement: pairwise traces and the three-density trace.
    expected = (2-sum(np.trace(rhos[i]@rhos[j]).real for i, j in ((0, 1), (1, 2), (0, 2)))
                +4*np.trace(rhos[0]@rhos[1]@rhos[2]).real)/9
    assert result.spatial_probabilities()[(1, 1, 1)] == pytest.approx(expected, abs=3e-14)
    assert result.state.trace == pytest.approx(1, abs=3e-14)


def test_mixed_states_keep_joint_basis_and_loss():
    a = np.array([1, 1j])/np.sqrt(2)
    b = np.array([np.sqrt(.3), np.sqrt(.7)])
    rhos = [np.outer(a, a.conj()), np.outer(b, b.conj())]
    circuit = Circuit(2).bs(0, 1, transmission=.37).loss(0, .8).loss(1, .6)
    mixed = mixed_wavepacket_input(2, [1, 0], rhos).through(circuit).spatial_probabilities()
    g = np.column_stack([a, b]).conj().T@np.column_stack([a, b])
    pure = wavepacket_input(2, [1, 0], g).through(circuit).spatial_probabilities()
    assert mixed == pytest.approx(pure, abs=3e-14)
    for ports, matrices in (([0, 0], rhos), ([0, 1], [rhos[0]]), ([0, 1], [rhos[0], np.eye(3)/3])):
        with pytest.raises(ValueError):
            mixed_wavepacket_input(2, ports, matrices)
    with pytest.raises(ResourceLimitError):
        mixed_wavepacket_input(2, [0, 1], rhos, precision=Precision(max_dimension=5))


def test_number_sources_reduce_to_fock_and_preserve_cutoff_tail():
    distributions = [[.2, .7, .1], [.3, .65, .05]]
    circuit = Circuit(2).bs(0, 1, transmission=.37).loss(0, .71)
    full = wavepacket_sources(2, [0, 1], distributions, np.ones((2, 2)),
                              multiphoton_model="same_wavepacket")
    expected = circuit.run(independent_sources(FockBasis(2, 4), distributions).state)
    assert full.through(circuit).spatial_probabilities() == pytest.approx(expected.probabilities(), abs=3e-14)
    cut = wavepacket_sources(2, [0, 1], distributions, [[1, .3], [.3, 1]],
                             multiphoton_model="orthogonal_noise", max_photons=2)
    omitted = sum(distributions[0][a]*distributions[1][b] for a, b in product(range(3), repeat=2) if a+b > 2)
    assert cut.source_omitted_probability == pytest.approx(omitted, abs=2e-15)
    assert cut.state.trace == pytest.approx(1-omitted, abs=2e-15)
    assert cut.through(circuit).source_omitted_probability == cut.source_omitted_probability
    with pytest.raises(ResourceLimitError):
        wavepacket_sources(2, [0, 1], distributions, np.eye(2),
                           multiphoton_model="same_wavepacket", max_source_patterns=3)
    with pytest.raises(ValueError):
        wavepacket_sources(1, [0], [[0, 0, 0, 1]], [[1]], multiphoton_model="orthogonal_noise")


def test_large_bosonic_product_avoids_factorial_overflow():
    state = wavepacket_input(1, [0]*200, np.ones((200, 200)))
    assert state.internal_modes == 1
    assert state.spatial_probabilities()[(200,)] == pytest.approx(1)
    source = wavepacket_sources(1, [0], [[0]*200+[1]], [[1]], multiphoton_model="same_wavepacket")
    assert source.state.trace == pytest.approx(1)
    with pytest.raises(ValueError, match="underflows"):
        number_distribution_from_moments(1e-200, .5)


@pytest.mark.parametrize("model", ["same_wavepacket", "orthogonal_noise"])
def test_source_hom_intensity_moments_against_operator_formula(model):
    pa, pb = np.array([.2, .7, .1]), np.array([.3, .65, .05])
    m, t, eta0, eta1 = .79, .37, .81, .72
    result = source_hom([pa, pb], m, multiphoton_model=model,
                        beamsplitter_transmission=t, transmissions=(eta0, eta1),
                        detectors=[Detector(threshold=True, efficiency=.61), Detector(threshold=True, efficiency=.87)])
    ma, mb = pa[1]+2*pa[2], pb[1]+2*pb[2]
    interference = m*(ma*mb if model == "same_wavepacket" else (pa[1]+pa[2])*(pb[1]+pb[2]))
    expected = eta0*eta1*(t*(1-t)*2*(pa[2]+pb[2])
                          +(t*t+(1-t)**2)*ma*mb-2*t*(1-t)*interference)
    assert result["parallel"]["intensity_cross_moment"] == pytest.approx(expected, abs=3e-14)
    assert result["parallel"]["total_probability"] == pytest.approx(1, abs=3e-14)
    assert result["parallel"]["source_omitted_probability"] == 0


def test_two_photon_sector_noise_changes_interference_and_zero_click_reference():
    # Exactly two photons in each port: indistinguishable HOM removes all odd
    # output occupations; source-specific orthogonal noise restores them.
    values = []
    for model in ("same_wavepacket", "orthogonal_noise"):
        state = wavepacket_sources(2, [0, 1], [[0, 0, 1]]*2, np.ones((2, 2)), multiphoton_model=model)
        p = state.through(Circuit(2).bs(0, 1)).spatial_probabilities()
        values.append(p.get((1, 3), 0)+p.get((3, 1), 0))
    assert values[0] == pytest.approx(0, abs=3e-14)
    assert values[1] == pytest.approx(.5, abs=3e-14)
    dark = source_hom([[1], [1]], 1, multiphoton_model="same_wavepacket")
    assert dark["click_visibility"] is None


def test_lossy_transfer_against_independent_halmos_vacuum_dilation():
    a = np.array([[.45+.1j, .2], [-.15j, .57-.04j]])
    dilation = np.block([[a, sqrtm(np.eye(2)-a@a.conj().T)],
                         [sqrtm(np.eye(2)-a.conj().T@a), -a.conj().T]])
    full = Circuit(4).unitary(dilation).run([1, 1, 0, 0]).state
    expected = sum(herald(full, [2, 3], [i, j]).remaining_state.rho
                   for i in range(3) for j in range(3-i))
    circuit = Circuit(2).transfer(a)
    actual = circuit.run([1, 1])
    assert np.allclose(actual.state.rho, expected, atol=3e-14, rtol=0)
    assert actual.state.trace == pytest.approx(1, abs=3e-14)
    assert actual.transfer_residuals[0]["field_matrix_residual"] < 1e-14
    assert np.allclose(circuit.channel(FockBasis(2, 2)).apply([0, 0, 0, 0, 1, 0]).rho,
                       actual.state.rho, atol=3e-14, rtol=0)
    photon = circuit.run([1, 0]).probabilities()
    assert photon[(1, 0)] == pytest.approx(abs(a[0, 0])**2)
    assert photon[(0, 1)] == pytest.approx(abs(a[1, 0])**2)
    assert photon[(0, 0)] == pytest.approx(1-sum(abs(a[:, 0])**2))
    with pytest.raises(ValueError):
        Circuit(2).transfer(np.eye(2)*1.01)


def test_source_hom_cli_and_ambiguous_fields(tmp_path, capsys):
    config = {"sources": [{"mean_photons": .154, "g2_zero": .0028}]*2,
              "indistinguishability": .9956, "multiphoton_model": "same_wavepacket"}
    path = tmp_path/"hom.json"
    output = tmp_path/"run.json"
    path.write_text(json.dumps(config))
    main(["source-hom", str(path), "--output", str(output)])
    payload = json.loads(output.read_text())
    assert 0 < payload["click_visibility"] < .9956
    main(["hom", "--indistinguishability", ".985"])
    assert json.loads(capsys.readouterr().out)["coincidence_probability"] == pytest.approx(.0075)
    with pytest.raises(SystemExit):
        main(["hom", "--overlap", ".7", "--indistinguishability", ".7"])
    with pytest.raises(ValueError):
        source_hom_scenario({**config, "sources": [{"probabilities": [0, 1], "g2_zero": 0}]*2})
    result = optical_scenario({"modes": 1, "input": {"occupation": [1]},
                               "steps": [{"operation": "transfer", "real": [[.5]]}]})
    assert result["transfer_residuals"][0]["field_matrix_residual"] < 1e-14
