"""Resource and support regressions from the 2026-09-07 execution review."""
import copy
import json

import pytest

from plq import ResourceLimitError, five_qubit_code
from plq.cli import main, memory_scenario, optical_scenario
from plq.qec import ErasureDecoder, MinimumWeightDecoder
from plq.simulation import MemoryNoise, simulate_memory


@pytest.mark.parametrize("decoder", [None, "minimum_weight", "erasure"])
def test_memory_cli_decoder_budget_applies_with_or_without_selection(decoder):
    config = {"code": {"name": "five_qubit"}, "shots": 1, "max_patterns": 1}
    if decoder is not None:
        config["decoder"] = decoder
    with pytest.raises(ResourceLimitError, match="max_patterns"):
        memory_scenario(config)


@pytest.mark.parametrize("erasure", [0., .1, [0., 0., .1, 0., 0.]])
def test_memory_python_default_decoder_honors_budget(erasure):
    with pytest.raises(ResourceLimitError, match="max_patterns"):
        simulate_memory(five_qubit_code(), MemoryNoise(erasure=erasure),
                        shots=1, max_patterns=1)


def test_default_erasure_decoder_enforces_budget_during_decode():
    # 16 patterns suffice to build the five-qubit fallback. An erasure table
    # requires 4**5=1024, so the budget must survive construction as well.
    with pytest.raises(ResourceLimitError, match="Erasure decoding exceeds"):
        memory_scenario({"code": {"name": "five_qubit"}, "shots": 1,
                         "noise": {"erasure": 1.}, "max_patterns": 16})


@pytest.mark.parametrize("erasure,decoder_type", [(0., MinimumWeightDecoder), (.1, ErasureDecoder)])
def test_default_decoder_keeps_seeded_results(erasure, decoder_type):
    code = five_qubit_code()
    noise = MemoryNoise(px=.05, erasure=erasure)
    options = {"shots": 50, "rounds": 2, "seed": 29}
    expected = simulate_memory(code, noise, decoder=decoder_type(code), **options)
    actual = simulate_memory(code, noise, max_patterns=1024, **options)
    assert actual.to_dict() == expected.to_dict()


@pytest.mark.parametrize("budget", [0, -1, 1.5, True])
def test_memory_python_rejects_invalid_budget(budget):
    with pytest.raises(ValueError, match="max_patterns"):
        simulate_memory(five_qubit_code(), shots=1, max_patterns=budget)


def test_custom_decoder_is_preserved_even_if_falsey():
    class IdentityDecoder:
        calls = 0

        def __bool__(self):
            return False

        def decode(self, syndrome, *, erasures=(), history=()):
            self.calls += 1
            return "IIIII"

    decoder = IdentityDecoder()
    result = simulate_memory(five_qubit_code(), shots=1, decoder=decoder, max_patterns=1)
    assert result.failures == 0
    assert decoder.calls == 2


@pytest.mark.parametrize("sources,cutoff", [([[1.], [1.]], 0),
    ([[.2, 0., .8], [.5, .5]], 3), ([[1., 1e-20], [1.]], 1)])
def test_source_padding_does_not_change_state_or_inferred_cutoff(sources, cutoff):
    config = {"modes": 2, "input": {"sources": sources},
              "steps": [{"operation": "bs", "first": 0, "second": 1}]}
    padded = copy.deepcopy(config)
    padded["input"]["sources"] = [dist + [0.] * 100 for dist in sources]
    snapshot = copy.deepcopy(padded)
    expected, actual = optical_scenario(config), optical_scenario(padded)
    assert padded == snapshot
    assert actual["initial_total_cutoff"] == cutoff
    for key in ("probabilities", "diagnostics", "source_omitted_probability", "trace_history"):
        assert actual[key] == expected[key]


def test_tiny_positive_tail_still_triggers_dimension_guard():
    with pytest.raises(ResourceLimitError, match="Hilbert dimension"):
        optical_scenario({"modes": 2, "input": {"sources": [
            [1.] + [0.] * 99 + [1e-20], [1.]]}})


@pytest.mark.parametrize("dist", [[], [0., 0.], [.8, 0.], [1., -1e-20],
                                  [1., float("nan")], [1., float("inf")]])
@pytest.mark.parametrize("cutoff", [None, 0])
def test_invalid_sources_are_rejected_with_auto_or_explicit_cutoff(dist, cutoff):
    config = {"modes": 2, "input": {"sources": [dist + [0.] * 100, [1.]]}}
    if cutoff is not None:
        config["max_photons"] = cutoff
    with pytest.raises(ValueError) as error:
        optical_scenario(config)
    assert not isinstance(error.value, ResourceLimitError)


def test_explicit_source_cutoff_still_reports_omitted_probability():
    result = optical_scenario({"modes": 2, "max_photons": 0,
        "input": {"sources": [[.75, .25] + [0.] * 100, [1.]]}})
    assert result["initial_total_cutoff"] == 0
    assert result["source_omitted_probability"] == pytest.approx(.25)
    assert result["diagnostics"]["trace_real"] == pytest.approx(.75)


def test_source_normalization_uses_configured_tolerance():
    config = {"modes": 1, "input": {"sources": [[1. - 1e-8, 0.]]}}
    with pytest.raises(ValueError, match="sum to one"):
        optical_scenario(config)
    result = optical_scenario({**config, "precision": {"atol": 1e-7}})
    assert result["source_omitted_probability"] == pytest.approx(1e-8)


def test_source_mode_count_validated_before_dimension_guard():
    with pytest.raises(ValueError, match="One number distribution"):
        optical_scenario({"modes": 2, "input": {"sources": [[0.] * 100 + [1.]]}})


def test_review_cases_through_actual_cli(tmp_path, capsys):
    config = tmp_path / "scenario.json"
    output = tmp_path / "result.json"
    config.write_text(json.dumps({"modes": 2, "input": {"sources": [
        [1.] + [0.] * 100, [1.] + [0.] * 100]}}))
    main(["optics", str(config), "--output", str(output)])
    assert json.loads(output.read_text())["initial_total_cutoff"] == 0
    config.write_text(json.dumps({"code": {"name": "five_qubit"}, "shots": 1, "max_patterns": 1}))
    with pytest.raises(SystemExit) as error:
        main(["memory", str(config)])
    assert error.value.code == 2
    assert "max_patterns" in capsys.readouterr().err
