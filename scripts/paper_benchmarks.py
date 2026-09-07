"""Run source-paper scenarios and independent interference formulas; no downloads."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import platform
import subprocess
import numpy as np
import scipy
from plq import (__version__, Circuit, Detector, mixed_wavepacket_input, wavepacket_input,
                 detection_probabilities)
from plq.cli import source_hom_scenario
from plq.reproduction import evidence_record


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT/path).read_text(encoding="utf-8"))


def source_cases(metadata):
    cases = []
    for key in ("somaschi_2016", "ding_2016"):
        paper = metadata[key]
        config = read(paper["scenario"])
        for model in ("same_wavepacket", "orthogonal_noise"):
            config = {**config, "multiphoton_model": model}
            result = source_hom_scenario(config)
            a, b = result["source_moments"]
            ma, mb = a["mean_photons"], b["mean_photons"]
            f2a, f2b = a["factorial_second_moment"], b["factorial_second_moment"]
            overlap = config["indistinguishability"]
            exchange = overlap*(ma*mb if model == "same_wavepacket" else (ma-f2a/2)*(mb-f2b/2))
            t = config["beamsplitter_transmission"]
            eta0, eta1 = config.get("transmissions", [1, 1])
            expected = eta0*eta1*(t*(1-t)*(f2a+f2b)+(t*t+(1-t)**2)*ma*mb-2*t*(1-t)*exchange)
            error = abs(result["parallel"]["intensity_cross_moment"]-expected)
            if error > 5e-12 or abs(result["parallel"]["total_probability"]-1) > 5e-12:
                raise AssertionError(f"Source moment or trace check failed: {key}/{model}")
            sensitivity = []
            for parameter in ("indistinguishability", "g2_zero"):
                measured = paper["reported"][parameter]
                for sign in (-1, 1):
                    value = max(0., measured["value"]+sign*measured["uncertainty"])
                    if parameter == "indistinguishability":
                        value = min(1., value)
                    varied = deepcopy(config)
                    if parameter == "indistinguishability":
                        varied[parameter] = value
                    else:
                        for source in varied["sources"]:
                            source[parameter] = value
                    evaluated = source_hom_scenario(varied)
                    sensitivity.append({"parameter": parameter, "direction": sign, "value": value,
                                        "click_visibility": evaluated["click_visibility"]})
            cases.append({"paper": key, "source_url": paper["url"], "result": result,
                          "independent_intensity_cross_moment": expected,
                          "intensity_moment_absolute_error": error,
                          "one_parameter_sensitivity": sensitivity})
    return cases


def ding_rate(metadata):
    p = metadata["ding_2016"]["reported"]
    names = ("preparation_efficiency", "extraction_efficiency", "polarization_transmission",
             "path_transmission", "fiber_coupling")
    circuit = Circuit(1)
    for name in names:
        circuit.loss(0, p[name])
    state = circuit.run([1]).state
    detector = Detector(threshold=True, efficiency=p["detector_efficiency"])
    click = detection_probabilities(state, detectors=[detector])[(1,)]
    predicted = click*p["repetition_rate_hz"]
    expected = float(np.prod([p[name] for name in names]))*p["detector_efficiency"]*p["repetition_rate_hz"]
    if abs(predicted-expected) > 1e-7:
        raise AssertionError("Sequential loss and count budget disagree")
    return {"simulated_rate_hz": predicted, "independent_rate_hz": expected,
            "reported_rate_hz": p["detected_rate_hz"],
            "relative_difference": predicted/p["detected_rate_hz"]-1,
            "interpretation": "Rounded pre-etalon efficiency budget; no parameter fitting or calibrated error bar."}


def interference_cases():
    u = np.exp(-2j*np.pi*np.outer(np.arange(3), np.arange(3))/3)/np.sqrt(3)
    circuit = Circuit(3).unitary(u)
    triad = []
    for phi in np.linspace(0, 2*np.pi, 9):
        gram = np.array([[1, .5, .5*np.exp(-1j*phi)], [.5, 1, .5], [.5*np.exp(1j*phi), .5, 1]])
        output = wavepacket_input(3, [0, 1, 2], gram).through(circuit)
        measured = output.spatial_probabilities()[(1, 1, 1)]
        expected = (2-.75+.5*np.cos(phi))/9
        error = abs(measured-expected)
        if error > 5e-12:
            raise AssertionError("Menssen Eq. (4) mismatch")
        triad.append({"triad_phase": float(phi), "p111": measured, "analytic_p111": float(expected),
                      "absolute_error": error})
    rho = np.diag([.7, .3])
    output = mixed_wavepacket_input(3, [0, 1, 2], [rho]*3).through(circuit)
    purity, third = float(np.trace(rho@rho).real), float(np.trace(rho@rho@rho).real)
    expected = (2-3*purity+4*third)/9
    actual = output.spatial_probabilities()[(1, 1, 1)]
    # Deliberately incorrect surrogate, used to quantify why retaining mixed
    # density matrices matters. It preserves pairwise HOM values only.
    fake_gram = np.full((3, 3), np.sqrt(purity))
    np.fill_diagonal(fake_gram, 1)
    fake = wavepacket_input(3, [0, 1, 2], fake_gram).through(circuit).spatial_probabilities()[(1, 1, 1)]
    if abs(actual-expected) > 5e-12:
        raise AssertionError("Mixed-state density invariant mismatch")
    mixed = {"chosen_internal_density_eigenvalues": [.7, .3], "pairwise_trace_overlap": purity,
             "three_density_trace": third, "p111": actual, "analytic_p111": expected,
             "absolute_error": abs(actual-expected), "incorrect_pairwise_only_surrogate_p111": fake,
             "interpretation": "Selected validation state, not a measured paper spectrum."}
    rare = []
    for overlap in (1-1e-12, 1-1e-14):
        result = wavepacket_input(2, [0, 1], [[1, overlap], [overlap, 1]])
        actual = result.through(Circuit(2).bs(0, 1)).spatial_probabilities()[(1, 1)]
        expected = (1-overlap)*(1+overlap)/2
        if abs(actual/expected-1) > 1e-10:
            raise AssertionError("Rare HOM probability lost relative accuracy")
        rare.append({"overlap_amplitude": overlap, "p11": actual, "analytic_p11": expected,
                     "relative_error": abs(actual/expected-1)})
    return triad, mixed, rare


def build_report():
    metadata = read("benchmarks/paper_parameters.json")
    triad, mixed, rare = interference_cases()
    provenance_paths = ["benchmarks/paper_parameters.json", "examples/papers/somaschi_2016.json",
                        "examples/papers/ding_2016.json", "scripts/paper_benchmarks.py"]
    provenance_paths += [str(p.relative_to(ROOT)) for p in sorted((ROOT/"src/plq").rglob("*.py"))]
    hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in provenance_paths}
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        head, dirty = None, None
    return {"evidence": evidence_record("paper_parameter_reproduction"), "schema": "plq.paper-benchmarks.v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
            "versions": {"plq": __version__, "python": platform.python_version(),
                         "numpy": np.__version__, "scipy": scipy.__version__},
            "source_provenance": {"checkout_head": head, "working_tree_dirty": dirty, "sha256": hashes},
            "references": metadata, "baseline": read("benchmarks/baseline_v02.json"),
            "source_cases": source_cases(metadata), "ding_rate_budget": ding_rate(metadata),
            "menssen_triad_sweep": triad, "mixed_state_check": mixed, "rare_hom": rare,
            "scope": "Source-summary sensitivity scenarios, a rounded count budget, and formula regressions. No full experimental reproduction or hardware calibration."}


def render(report):
    lines = ["# Paper-parameter benchmarks", "", f"Recorded: {report['generated_utc']}", "",
             f"PLQ {report['versions']['plq']}. All numbers below were computed by `scripts/paper_benchmarks.py`.", "",
             "Evidence category: `paper_parameter_reproduction`; `experimental_reproduction=false`. See [the comparison contract](REPRODUCTION.md).", "",
             "Three papers supply numerical scenarios; a fourth motivates explicit noise hypotheses. "
             "Published measurements, derived inputs and scenario assumptions are recorded separately in "
             "[paper_parameters.json](../benchmarks/paper_parameters.json). Full machine results, configurations, "
             "versions and input hashes are in [paper_benchmarks.json](../benchmarks/paper_benchmarks.json).", "",
             "## Before and after", "",
             "The unchanged v0.2 source was executed from commit `c4914874603ee7a26ed2d77973f20a8b10eda1e9`. "
             "It already agrees with the pure-state single-photon and triad formulas. The concrete regression "
             "was its numerical rank cutoff: positive Gram eigenvalues below `atol` disappeared. "
             "The new two-wavepacket factorization also avoids subtractive eigensolver error near unit overlap.", "",
             "| Overlap amplitude | v0.2 P(1,1) | v0.3 P(1,1) | Analytic reference |",
             "| --- | --- | --- | --- |"]
    for before, after in zip(report["baseline"]["rare_hom"], report["rare_hom"]):
        lines.append(f"| {after['overlap_amplitude']:.15g} | {before['probability']:.9g} | {after['p11']:.12g} | {after['analytic_p11']:.12g} |")
    lines += ["", "These are floating-point regression cases, not demonstrated experimental precision.", "",
              "## Source scenarios", "",
              "[Somaschi et al.](https://arxiv.org/abs/1510.06499) supplies brightness 0.154, g2=0.0028 and "
              "corrected overlap 0.9956. This combines abstract summaries; the body/caption does not establish "
              "one jointly calibrated operating point for the triplet.", "",
              "[Ding et al.](https://arxiv.org/abs/1601.00284) supplies overlap 0.985 and g2=0.009. "
              "Its scenario mean 0.6336 is derived from preparation*extraction, and 0.216 from three downstream "
              "efficiencies. The etalon used for purity/HOM measurements is not included in the rate budget, "
              "so this combination is a sensitivity scenario.", "",
              "Each run keeps vacuum through four-photon sectors, the supplied beam splitter, losses and threshold "
              "detection. `same_wavepacket` puts extra photons in the source's signal mode. `orthogonal_noise` "
              "uses a source-specific orthogonal extra photon only in P(2). These hypotheses are neither "
              "statistical bounds nor uniquely determined by g2.", "",
              "| Paper | Extra-photon hypothesis | Absolute click coincidence | Click visibility | Moment-formula error |",
              "| --- | --- | --- | --- | --- |"]
    for case in report["source_cases"]:
        r = case["result"]
        lines.append(f"| {case['paper']} | {r['multiphoton_model']} | {r['parallel']['coincidence_probability']:.10g} | {r['click_visibility']:.10g} | {case['intensity_moment_absolute_error']:.3g} |")
    lines += ["", "Click visibility here is 1-C_parallel/C_distinguishable for direct two-input experiments. "
              "It is not the papers' corrected overlap or their pulsed-interferometer histogram normalization. "
              "Reported uncorrected values are retained as context, not used as equal-observable fit targets.", "",
              "The JSON also reruns each scenario while varying M and g2 separately by their quoted uncertainties, "
              "clipping M to its physical interval. This is local sensitivity, not a confidence interval: "
              "joint calibration, covariance and a likelihood are unavailable.", "",
              "[Ollivier et al.](https://arxiv.org/abs/2005.01743) demonstrates why extra-photon overlap matters. "
              "Its weak separable-noise approximation has a different source construction and observable; "
              "the benchmark does not equate it with the conditional P(2) models above.", "",
              "## Measured count-budget comparison", ""]
    rate = report["ding_rate_budget"]
    lines += [f"Sequential pure-loss channels plus a threshold detector give **{rate['simulated_rate_hz']:,.3f} counts/s**, "
              f"against **{rate['reported_rate_hz']:,.0f} counts/s** reported by Ding et al. "
              f"The relative difference is **{100*rate['relative_difference']:.4f}%**. "
              "The input factors are independently rounded; no fitting or uncertainty bound is claimed. "
              "This check pertains to the pre-etalon count budget.", "",
              "## Three-photon interference", "",
              "[Menssen et al.](https://arxiv.org/abs/1609.09804), Eq. (4), predicts "
              "P111=(2-3r^2+4r^3 cos(phi))/9 for equal overlap moduli r. "
              "The r=0.5 scan holds every pairwise HOM overlap fixed.", "",
              "| Triad phase / pi | Simulated P111 | Formula P111 | Absolute error |",
              "| --- | --- | --- | --- |"]
    for r in report["menssen_triad_sweep"]:
        lines.append(f"| {r['triad_phase']/np.pi:.2f} | {r['p111']:.12g} | {r['analytic_p111']:.12g} | {r['absolute_error']:.3g} |")
    m = report["mixed_state_check"]
    lines += ["", f"For a selected mixed internal state diag(0.7,0.3), the new density-matrix input gives "
              f"**P111={m['p111']:.12g}**, matching the supplement's density-trace formula. "
              f"Replacing the state by pure packets with the same pairwise HOM overlaps gives "
              f"**{m['incorrect_pairwise_only_surrogate_p111']:.12g}** instead. "
              "The selected eigenvalues are a verification example, not a measured spectrum.", "",
              "## Reproduce", "", "```bash", 'python -m pip install -e ".[test,reference]"',
              "python -m plq source-hom examples/papers/somaschi_2016.json",
              "python -m plq source-hom examples/papers/ding_2016.json --output results/ding.json",
              "python examples/mixed_wavepackets.py", "OPENBLAS_NUM_THREADS=1 python scripts/paper_benchmarks.py", "```", "",
              "Independent tests additionally compare lossy transfer matrices with a full vacuum-environment "
              "unitary dilation, and check mixed-state coherences, source cutoffs and photon-number moments. "
              "These checks establish implementation agreement within the stated models; they do not validate "
              "unmodeled spectral correlations, detector recovery, higher source sectors or entire hardware systems.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Run comparisons without rewriting reports")
    args = parser.parse_args()
    report = build_report()
    if not args.check:
        (ROOT/"benchmarks/paper_benchmarks.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        (ROOT/"docs/PAPER_BENCHMARKS.md").write_text(render(report), encoding="utf-8")
    print(json.dumps({"source_scenarios": len(report["source_cases"]),
                      "source_sensitivity_runs": sum(len(x["one_parameter_sensitivity"]) for x in report["source_cases"]),
                      "max_triad_error": max(x["absolute_error"] for x in report["menssen_triad_sweep"]),
                      "ding_rate_relative_difference": report["ding_rate_budget"]["relative_difference"]}, indent=2))


if __name__ == "__main__":
    main()
