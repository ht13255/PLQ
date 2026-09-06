from itertools import product
from math import comb, fsum
import json
import numpy as np
import pytest
from plq import (
    Circuit, Detector, DualRail, FockBasis, FockState, Precision, ResourceLimitError,
    MaximumLikelihoodDecoder, MinimumWeightDecoder, MemoryNoise, StabilizerCode,
    exact_pauli_memory, repetition_code, five_qubit_code, herald, simulate_memory,
)
from plq.qec import parse_pauli
from plq.thermal import ThermalBath
from plq.cli import main, optical_scenario, memory_scenario


def test_detector_rare_events_are_not_subtracted_from_one():
    assert Detector(threshold=True, dark_rate_hz=1e-11).response(0)[1] == pytest.approx(1e-20, rel=1e-14, abs=0)
    assert Detector(threshold=True, efficiency=1e-20).response(2)[1] == pytest.approx(2e-20, rel=1e-14, abs=0)
    assert np.array_equal(Detector(threshold=True).response(0), [1, 0])
    assert np.array_equal(Detector(threshold=True).response(1), [0, 1])


def test_detector_far_tail_against_high_precision_integral():
    mp = pytest.importorskip("mpmath")
    detector = Detector(gate_seconds=.02, arrival_offset_seconds=9, timing_sigma_seconds=1)
    with mp.workdps(60):
        expected = float(mp.quad(lambda x: mp.exp(-x*x/2)/mp.sqrt(2*mp.pi), [mp.mpf('8.99'), mp.mpf('9.01')]))
    assert detector.effective_efficiency == pytest.approx(expected, rel=5e-12, abs=0)


@pytest.mark.parametrize("n,eta,mu", [(5000, .4, 2.), (10000, .0001, .3), (0, .2, 200.), (7, 1., 0.)])
def test_large_photon_detector_is_complete(n, eta, mu):
    detector = Detector(efficiency=eta, dark_rate_hz=mu, gate_seconds=1, saturation=4)
    p = detector.response(n)
    assert np.all(np.isfinite(p)) and np.all(p >= 0)
    assert p.sum() == pytest.approx(1, abs=2e-12)
    if n == 10000:
        # Independently evaluate the low-count convolution (small binomial k).
        from math import exp, factorial
        for k in range(4):
            expected = fsum(comb(n, j)*eta**j*(1-eta)**(n-j)*exp(-mu)*mu**(k-j)/factorial(k-j)
                            for j in range(k+1))
            assert p[k] == pytest.approx(expected, rel=2e-11, abs=0)


@pytest.mark.parametrize("eta", [0., .37, 1.])
def test_thermal_vacuum_bath_reduces_to_pure_loss(eta):
    basis = FockBasis(2, 2)
    state = FockState.amplitudes(basis, {(0, 0): .6, (2, 0): .8j})
    actual = Circuit(2).thermal_loss(0, eta, 0).run(state)
    expected = Circuit(2).loss(0, eta).run(state)
    assert np.allclose(actual.state.rho, expected.state.rho, atol=2e-13, rtol=0)
    assert actual.bath_omitted_probability == 0


def test_thermal_vacuum_output_and_reported_geometric_tail():
    mean, eta, cutoff = .2, .7, 12
    result = Circuit(1).thermal_loss(0, eta, mean, bath_cutoff=cutoff).run([0])
    tail = (mean/(1+mean))**(cutoff+1)
    assert result.bath_omitted_probability == pytest.approx(tail, rel=1e-13, abs=0)
    assert result.state.trace == pytest.approx(1-tail, abs=3e-14)
    assert result.numerical_trace_error < 3e-14
    output_mean = (1-eta)*mean
    exact = np.array([output_mean**n/(1+output_mean)**(n+1) for n in range(cutoff+1)])
    actual = np.diag(result.state.rho).real
    # Missing bath branches are positive; their trace bounds any count event.
    assert np.max(abs(actual-exact)) <= tail+3e-14
    observed_mean = sum(n[0]*p for n, p in result.probabilities().items())
    exact_retained_mean = output_mean-(1-eta)*(cutoff+1+mean)*tail
    assert observed_mean == pytest.approx(exact_retained_mean, abs=3e-14)


def test_thermal_dilation_against_independent_full_fock_recurrence():
    eta, mean, cutoff = .63, .4, 2
    state = FockState.amplitudes(FockBasis(2, 1), {(0, 0): .6, (1, 0): .48j, (0, 1): .64})
    actual = Circuit(2).thermal_loss(0, eta, mean, bath_cutoff=cutoff).run(state).state
    # Build the complete system + bath density matrix independently, lift the
    # ordinary passive unitary by creation recurrence, then trace the bath.
    basis = FockBasis(3, 1+cutoff)
    rho = np.zeros((basis.dimension, basis.dimension), complex)
    for k in range(cutoff+1):
        weight = mean**k/(1+mean)**(k+1)
        indices = [basis.index[(*occ, k)] for occ in state.basis.states]
        rho[np.ix_(indices, indices)] += weight*state.rho
    full = Circuit(3).bs(0, 2, transmission=eta).run(FockState(basis, rho, subnormalized=True)).state
    reduced = sum(herald(full, [2], [k]).remaining_state.rho for k in range(1+cutoff+1))
    assert np.allclose(actual.rho, reduced, atol=3e-13, rtol=0)


def test_thermal_basis_grows_between_layers_and_kraus_matches():
    basis = FockBasis(1, 1)
    state = FockState.amplitudes(basis, {(0,): .6, (1,): .8j})
    circuit = (Circuit(1).thermal_loss(0, .6, .3, bath_cutoff=2)
               .phase(0, .43).loss(0, .8).thermal_loss(0, .4, .2, bath_cutoff=1))
    result = circuit.run(state)
    channel = circuit.channel(basis)
    assert channel.output_basis.max_photons == result.state.basis.max_photons == 4
    assert not channel.trace_preserving
    assert np.allclose(channel.apply(state.rho).rho, result.state.rho, atol=3e-13, rtol=0)
    retained = (1-(.3/1.3)**3)*(1-(.2/1.2)**2)
    assert result.state.trace == pytest.approx(retained, abs=3e-13)
    assert result.bath_omitted_probability == pytest.approx(1-retained)


def test_thermal_identity_and_subnormalized_input():
    state = FockState.mixture(FockBasis(1, 1), {(1,): .3}, subnormalized=True)
    identity = Circuit(1).thermal_loss(0, 1, .5, bath_cutoff=2).run(state)
    assert np.array_equal(identity.state.rho, state.rho)
    assert identity.bath_omitted_probability == 0
    result = Circuit(1).thermal_loss(0, 0, .5, bath_cutoff=2).run(state)
    assert result.bath_omitted_probability == pytest.approx(.3*(1/3)**3)
    assert result.probabilities()[(0,)] == pytest.approx(.3/1.5)


def test_thermal_dualrail_channel_keeps_false_survival_and_tail():
    rails = DualRail(1)
    circuit = Circuit(2).thermal_loss(0, .5, .2, bath_cutoff=2)
    state = circuit.run(rails.encode([1, 0])).state
    expected = rails.decode(state)
    actual = rails.effective_channel(circuit).apply([1, 0])
    assert np.allclose(actual.rho, expected.rho, atol=3e-13, rtol=0)


def test_thermal_cutoff_validation_and_resource_budget():
    bath = ThermalBath.create(.5, tail_tolerance=1e-13)
    assert bath.omitted_probability <= 1e-13
    assert (.5/1.5)**bath.cutoff > 1e-13
    for kwargs in ({"mean_photons": -1}, {"mean_photons": .2, "bath_cutoff": -1},
                   {"mean_photons": .2, "tail_tolerance": 0}):
        with pytest.raises(ValueError):
            Circuit(1).thermal_loss(0, .5, **kwargs)
    with pytest.raises(ResourceLimitError):
        Circuit(2).thermal_loss(0, .5, 1, bath_cutoff=30).run([1, 1], precision=Precision(max_dimension=100))


@pytest.mark.parametrize("p", [0., 1e-12, .08, .49])
def test_exact_repetition_rate_including_rare_logical_failures(p):
    result = exact_pauli_memory(repetition_code(), MemoryNoise(px=p))
    assert result.logical_error_rate == pytest.approx(3*p*p-2*p**3, rel=2e-13, abs=0)
    assert result.total_probability == pytest.approx(1, abs=2e-14)


def test_biased_decoder_uses_likelihood_instead_of_weight():
    code, noise = repetition_code(), MemoryNoise(px=[.2, .2, .0001])
    weight = exact_pauli_memory(code, noise, decoder=MinimumWeightDecoder(code))
    likelihood = exact_pauli_memory(code, noise)
    assert weight.logical_error_rate == pytest.approx(.040032)
    assert likelihood.logical_error_rate == pytest.approx(.0001)


def test_coset_decoder_against_independent_stabilizer_group_sum():
    code = StabilizerCode(["ZZII", "IIZZ"])
    # Deliberately nonuniform probabilities, two logical qubits, and degeneracy.
    noise = MemoryNoise(px=[.1, .02, .16, .2], py=[.13, .18, .04, .06], pz=[.3, .3, .2, .1])
    decoder = MaximumLikelihoodDecoder(code, noise)
    p = np.column_stack((1-np.array(noise.px)-noise.py-noise.pz, noise.px, noise.py, noise.pz))
    group = [(0, 0)]
    for gx, gz in code.masks:
        group += [(x ^ gx, z ^ gz) for x, z in group]
    masses = {}
    for letters in product(range(4), repeat=code.n):
        _, _, x, z = parse_pauli("".join("IXYZ"[e] for e in letters))
        key = min((x ^ gx, z ^ gz) for gx, gz in group)
        masses[key] = masses.get(key, 0.)+np.prod([p[q, e] for q, e in enumerate(letters)])
    for syndrome in product((0, 1), repeat=len(code.generators)):
        _, _, cx, cz = parse_pauli(decoder.decode(syndrome))
        chosen = min((cx ^ gx, cz ^ gz) for gx, gz in group)
        optimum = max(v for (x, z), v in masses.items() if code.syndrome_masks(x, z) == syndrome)
        assert masses[chosen] == pytest.approx(optimum, rel=3e-14)


def test_maximum_likelihood_corrects_flagged_replacement_and_bounds_cache():
    code = five_qubit_code()
    decoder = MaximumLikelihoodDecoder(code, MemoryNoise(), max_cache_entries=2)
    for erased in [(0, 1), (1, 2), (2, 4)]:
        for letters in product("IXYZ", repeat=2):
            word = ["I"]*5
            for q, e in zip(erased, letters):
                word[q] = e
            _, _, x, z = parse_pauli("".join(word))
            _, _, cx, cz = parse_pauli(decoder.decode(code.syndrome_masks(x, z), erasures=erased))
            assert code.in_stabilizer(x ^ cx, z ^ cz)
    assert len(decoder._cache) <= 2


def test_exact_decoder_rejects_unmodelled_noise_and_excess_cost():
    code = repetition_code()
    for noise in (MemoryNoise(erasure=.1), MemoryNoise(syndrome_flip=.1)):
        with pytest.raises(ValueError):
            exact_pauli_memory(code, noise)
    with pytest.raises(ValueError):
        MaximumLikelihoodDecoder(code, MemoryNoise(syndrome_flip=.1))
    with pytest.raises(ResourceLimitError):
        exact_pauli_memory(code, MemoryNoise(px=.1, py=.1, pz=.1), max_patterns=10)
    with pytest.raises(ValueError):
        simulate_memory(code, shots=1, final_perfect_round="false")
    with pytest.raises(ValueError):
        MemoryNoise(px=np.array([.1+.2j]*3)).arrays(3, 2)


def test_simple_api_and_inferred_configuration_cutoff(tmp_path, capsys):
    result = Circuit(2).bs(0, 1).run([1, 1])
    assert result.probabilities()[(1, 1)] == pytest.approx(0, abs=1e-14)
    config = {"modes": 1, "input": {"occupation": [0]},
              "steps": [{"operation": "thermal_loss", "mode": 0, "transmission": .5,
                         "mean_photons": .2, "bath_cutoff": 3}]}
    data = optical_scenario(config)
    assert data["initial_total_cutoff"] == 0 and data["final_total_cutoff"] == 3
    assert data["bath_omitted_probability"] > 0
    memory = {"code": {"name": "repetition"}, "noise": {"px": .08}, "method": "exact"}
    assert memory_scenario(memory)["logical_error_rate"] == pytest.approx(.018176)
    with pytest.raises(ValueError):
        memory_scenario({**memory, "shots": 10})
    with pytest.raises(ValueError):
        optical_scenario([])
    codefile = tmp_path/"my-code.json"
    repetition_code().save(codefile)
    configfile = tmp_path/"memory.json"
    configfile.write_text(json.dumps({**memory, "code": {"file": "my-code.json"}}))
    main(["memory", str(configfile)])
    assert json.loads(capsys.readouterr().out)["logical_error_rate"] == pytest.approx(.018176)
    main(["doctor"])
    assert json.loads(capsys.readouterr().out)["packages"]["numpy"]["available"]
