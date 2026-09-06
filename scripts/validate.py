"""Regenerate measured validation; --core needs only .[test,reference]."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
from plq import (
    __version__, Circuit, Detector, FockBasis, FockState, MemoryNoise,
    MinimumWeightDecoder, exact_pauli_memory, five_qubit_code,
    repetition_code, simulate_memory, wavepacket_input,
)
from plq.optics import lift_unitary
from plq.reference import fock_amplitude


PACKAGES = {"numpy": "numpy", "scipy": "scipy", "pytest": "pytest", "mpmath": "mpmath",
            "perceval-quandela": "perceval", "pennylane": "pennylane",
            "stim": "stim", "pymatching": "pymatching"}


def package_versions():
    result = {"plq-sim": __version__, "python": platform.python_version()}
    for distribution, module in PACKAGES.items():
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            try:
                result[distribution] = getattr(importlib.import_module(module), "__version__", "source checkout")
            except ImportError:
                result[distribution] = None
    return result


def render_report(report):
    tests, checks = report["tests"], report["deterministic_checks"]
    lines = ["# Validation report", "", f"Recorded: {report['generated_utc']}", "",
             f"PLQ {__version__}; mode: **{report['mode']}**.", "",
             f"**{tests['passed']} passed; {tests['failures']} failures, {tests['errors']} errors, "
             f"{tests['skipped']} skipped; {tests['tests']} collected.**", "",
             "The report separates executed checks from unavailable optional SDK checks. "
             "Skipped integrations are not counted as passes.", "",
             "## Deterministic comparisons", "", "| Check | Measured value |", "| --- | --- |"]
    for name, value in checks.items():
        rendered = f"{value:.16g}" if isinstance(value, float) else str(value)
        lines.append(f"| {name.replace('_', ' ')} | {rendered} |")
    lines += ["", "The thermal implementation is also tested against an independent full system-plus-bath "
              "creation-operator Fock calculation, including coherences and an untouched spectator mode. "
              "Other regression cases cover cutoff growth across layers, rectangular Kraus maps, vacuum/identity limits, "
              "rare detector tails, saturated count distributions and explicit resource limits.", "",
              "The maximum-likelihood decoder is tested against independently enumerated stabilizer-group sums, "
              "including multiple logical qubits, site-dependent noise and flagged replacements. "
              "All previously supported core examples and code-correction tests are rerun.", "",
              "## Exact one-round decoder comparison", "",
              "Five-qubit code; independent per-site p(X)=0.001, p(Y)=0.001, p(Z)=0.15; "
              "ideal syndrome measurement and recovery. All 1,024 Pauli patterns are included.", "",
              "| Decoder | Logical block error probability |", "| --- | --- |"]
    for result in report["exact_decoder_comparison"]:
        lines.append(f"| {result['decoder']} | {result['logical_error_rate']:.14g} |")
    lines += ["", "These are different recovery choices under the same finite noise model. "
              "They are not hardware error rates or thresholds.", "",
              "## Thermal cutoff convergence", "",
              "Vacuum input, transmission 0.7, bath mean 0.2. The infinite-bath vacuum probability is 1/1.06.", "",
              "| Bath cutoff | Omitted weight | Vacuum-probability error | Numerical trace drift |",
              "| --- | --- | --- | --- |"]
    for row in report["thermal_convergence"]:
        lines.append(f"| {row['cutoff']} | {row['omitted_probability']:.6g} | {row['vacuum_error']:.6g} | {row['trace_drift']:.6g} |")
    memory = report["repetition_memory"]
    lines += ["", "## Sampled memory", "",
              f"Repetition code, p(X)=0.08, one round: {memory['failures']} failures / {memory['shots']} trials, "
              f"rate {memory['logical_error_rate']:.8f}, Wilson 95% interval {memory['wilson_95']}. "
              "The analytic rate is 0.018176. Seed 77.", ""]
    if report["stim_surface_memory"] is not None:
        result = report["stim_surface_memory"]
        lines += [f"Stim distance-3 rotated memory, three rounds, depolarization and measurement flips 0.008: "
                  f"{result['failures']} / {result['shots']} failures, seed 123. This is a separate circuit model.", ""]
    else:
        lines += ["Stim/PyMatching sampled comparison was not executed in core mode.", ""]
    lines += ["## Versions", "", "| Package | Version |", "| --- | --- |"]
    lines += [f"| {name} | {version or 'not installed'} |" for name, version in report["versions"].items()]
    lines += ["", "## Reproduce", "", "```bash", 'python -m pip install -e ".[test,reference]"',
              "python scripts/validate.py --core", "", "# Include actual optional SDK comparisons:",
              'python -m pip install -e ".[all]"', "python scripts/validate.py", "```", "",
              "The script sets a single BLAS thread for its pytest subprocess, regenerates this report, "
              "`benchmarks/validation.json`, and the direct validated-version list. The full command fails if any tests skip.", "",
              "Main arithmetic is complex128; the 70-decimal permanent is an independent reference only. "
              "Measured deviations are finite test-case results, not global numerical bounds or hardware calibration. "
              "Bath omissions, floating-point drift and Monte Carlo intervals quantify different errors.", "",
              "Hosted runs and their source commits are recorded in [GitHub Actions](https://github.com/ht13255/PLQ/actions). "
              "This report describes its generating environment; a local run is not evidence of hosted CI completion.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", action="store_true", help="Allow missing optional SDKs and report skipped integrations")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    versions = package_versions()
    required = ["pytest", "mpmath"] + ([] if args.core else ["perceval-quandela", "pennylane", "stim", "pymatching"])
    missing = [name for name in required if versions[name] is None]
    if missing:
        parser.error(f"Missing packages: {', '.join(missing)}. Install .[all], or .[test,reference] for --core.")
    with tempfile.TemporaryDirectory(prefix="plq-validation-") as folder:
        xml = Path(folder)/"pytest.xml"
        env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        subprocess.run([sys.executable, "-m", "pytest", "-q", f"--junitxml={xml}"], cwd=root, env=env, check=True)
        suites = list(ET.parse(xml).getroot().iter("testsuite"))
        tests = {key: sum(int(s.attrib.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
        tests["passed"] = tests["tests"]-tests["failures"]-tests["errors"]-tests["skipped"]
        tests["elapsed_seconds"] = sum(float(s.attrib.get("time", 0)) for s in suites)
    if not args.core and tests["skipped"]:
        raise RuntimeError("Full validation requires all integrations; tests were skipped")
    hom = Circuit(2).bs(0, 1).run([1, 1]).probabilities()
    rng = np.random.default_rng(3)
    u, _ = np.linalg.qr(rng.normal(size=(3, 3))+1j*rng.normal(size=(3, 3)))
    basis = FockBasis(3, 3)
    lifted = lift_unitary(u, basis)
    permanent_error = max(abs(lifted[i, j]-complex(fock_amplitude(u, s, t, digits=70)))
                          for i, t in enumerate(basis.states) for j, s in enumerate(basis.states))
    partial = wavepacket_input(2, [0, 1], [[1, .7], [.7, 1]]).through(Circuit(2).bs(0, 1)).spatial_probabilities()
    checks = {"hom_coincidence": hom[(1, 1)], "partial_hom_coincidence": partial[(1, 1)],
              "partial_hom_expected": .255, "max_fock_amplitude_error_vs_70_decimal_permanent": permanent_error,
              "rare_dark_click_probability": float(Detector(threshold=True, dark_rate_hz=1e-11).response(0)[1]),
              "detector_5000_photons_probability_sum": float(Detector(efficiency=.4).response(5000).sum())}
    code, noise = five_qubit_code(), MemoryNoise(px=.001, py=.001, pz=.15)
    comparison = [exact_pauli_memory(code, noise, decoder=MinimumWeightDecoder(code)).to_dict(),
                  exact_pauli_memory(code, noise).to_dict()]
    convergence = []
    for cutoff in (4, 8, 12):
        result = Circuit(1).thermal_loss(0, .7, .2, bath_cutoff=cutoff).run([0])
        convergence.append({"cutoff": cutoff, "omitted_probability": result.bath_omitted_probability,
                            "vacuum_error": abs(result.probabilities()[(0,)]-1/1.06),
                            "trace_drift": result.numerical_trace_error})
    memory = simulate_memory(repetition_code(), MemoryNoise(px=.08), shots=30000, seed=77)
    surface_result = None
    if not args.core:
        import perceval as pcvl
        import pennylane as qml
        import stim
        from plq.adapters.perceval import from_perceval, reference_probabilities
        from plq.adapters.pennylane import from_operations
        from plq.adapters.stim import simulate_stim
        design = pcvl.Circuit(3).add((0, 1), pcvl.BS.H()).add(1, pcvl.PS(.37)).add((1, 2), pcvl.BS.Ry(theta=.63))
        local = from_perceval(design)
        sdk_error = 0.
        for source in [(1, 1, 0), (2, 0, 1), (0, 3, 0)]:
            ours = local.run(FockState.ket(basis, source)).probabilities()
            ref = reference_probabilities(local, source)
            sdk_error = max(sdk_error, max(abs(p-ref.get(s, 0)) for s, p in ours.items()))
        wires = ["second", "first"]
        def algorithm():
            qml.RY(.61, wires="first")
            qml.CNOT(wires=["first", "second"])
            qml.AmplitudeDamping(.13, wires="second")
            qml.PhaseFlip(.2, wires="first")
        channel = from_operations(qml.tape.make_qscript(algorithm)().operations, wire_order=wires)
        @qml.qnode(qml.device("default.mixed", wires=wires))
        def reference_algorithm():
            algorithm()
            return qml.state()
        checks["max_probability_error_vs_perceval_slos"] = sdk_error
        checks["max_density_entry_error_vs_pennylane"] = float(np.max(abs(channel.apply([1, 0, 0, 0]).rho-reference_algorithm())))
        surface = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3,
                                         after_clifford_depolarization=.008, before_measure_flip_probability=.008)
        surface_result = simulate_stim(surface, shots=2000, seed=123).to_dict()
    report = {"schema": "plq.validation.v2", "generated_utc": datetime.now(timezone.utc).isoformat(),
              "mode": "core" if args.core else "full", "versions": versions, "tests": tests,
              "platform": {"system": platform.system(), "machine": platform.machine()},
              "deterministic_checks": checks, "thermal_convergence": convergence,
              "exact_decoder_comparison": comparison, "repetition_memory": memory.to_dict(),
              "stim_surface_memory": surface_result}
    (root/"benchmarks").mkdir(exist_ok=True)
    (root/"benchmarks/validation.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (root/"docs/VALIDATION.md").write_text(render_report(report), encoding="utf-8")
    lines = [f"# Direct versions from {report['mode']} validation; Python {versions['python']}.",
             "# Not a transitive lockfile; missing optional SDKs are omitted."]
    lines += [f"{name}=={versions[name]}" for name in PACKAGES if versions[name] is not None]
    (root/"requirements-validated.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    # An exact machine-readable record is also available in CI logs.
    print("PLQ_VALIDATION_JSON="+json.dumps(report, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
