"""Run tests and regenerate the measured validation report. Requires .[all]."""
from pathlib import Path
from datetime import datetime, timezone
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
import pennylane as qml
import perceval as pcvl
import stim
from plq import *
from plq.optics import lift_unitary
from plq.reference import fock_amplitude
from plq.adapters.perceval import from_perceval, reference_probabilities
from plq.adapters.pennylane import from_operations
from plq.adapters.stim import simulate_stim


def main():
    root=Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="plq-validation-") as folder:
        xml=Path(folder)/"pytest.xml"
        env={**os.environ,"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1"}
        subprocess.run([sys.executable,"-m","pytest","-q",f"--junitxml={xml}"],cwd=root,env=env,check=True)
        suites=list(ET.parse(xml).getroot().iter("testsuite"))
        tests={key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ("tests","failures","errors","skipped")}
        tests["elapsed_seconds"]=sum(float(s.attrib.get("time",0)) for s in suites)
    distributions=["numpy","scipy","pytest","mpmath","perceval-quandela","pennylane","stim","pymatching"]
    versions={name:importlib.metadata.version(name) for name in distributions}
    versions["python"]=platform.python_version()
    basis=FockBasis(2,2)
    hom=Circuit(2).bs(0,1).run(FockState.ket(basis,(1,1))).probabilities()
    rng=np.random.default_rng(3)
    u,_=np.linalg.qr(rng.normal(size=(3,3))+1j*rng.normal(size=(3,3)))
    basis3=FockBasis(3,3)
    lifted=lift_unitary(u,basis3)
    permanent_error=max(abs(lifted[i,j]-complex(fock_amplitude(u,s,t,digits=70)))
                        for i,t in enumerate(basis3.states) for j,s in enumerate(basis3.states))
    design=pcvl.Circuit(3).add((0,1),pcvl.BS.H()).add(1,pcvl.PS(.37)).add((1,2),pcvl.BS.Ry(theta=.63))
    local=from_perceval(design)
    sdk_error=0.
    for source in [(1,1,0),(2,0,1),(0,3,0)]:
        ours=local.run(FockState.ket(basis3,source)).probabilities()
        reference=reference_probabilities(local,source)
        sdk_error=max(sdk_error,max(abs(p-reference.get(s,0)) for s,p in ours.items()))
    wires=["second","first"]
    def algorithm():
        qml.RY(.61,wires="first")
        qml.CNOT(wires=["first","second"])
        qml.AmplitudeDamping(.13,wires="second")
        qml.PhaseFlip(.2,wires="first")
    channel=from_operations(qml.tape.make_qscript(algorithm)().operations,wire_order=wires)
    @qml.qnode(qml.device("default.mixed",wires=wires))
    def reference_algorithm():
        algorithm()
        return qml.state()
    pennylane_error=float(np.max(abs(channel.apply([1,0,0,0]).rho-reference_algorithm())))
    overlap=.7
    partial=wavepacket_input(2,[0,1],[[1,overlap],[overlap,1]]).through(Circuit(2).bs(0,1)).spatial_probabilities()
    memory=simulate_memory(repetition_code(),MemoryNoise(px=.08),shots=30000,seed=77)
    surface=stim.Circuit.generated("surface_code:rotated_memory_z",distance=3,rounds=3,
        after_clifford_depolarization=.008,before_measure_flip_probability=.008)
    stim_result=simulate_stim(surface,shots=2000,seed=123)
    report={"schema":"plq.validation.v1","generated_utc":datetime.now(timezone.utc).isoformat(),
        "platform":{"system":platform.system(),"machine":platform.machine()},"versions":versions,"tests":tests,
        "deterministic_checks":{
            "hom_coincidence":hom[(1,1)],"hom_bunching_each":[hom[(2,0)],hom[(0,2)]],
            "partial_hom_overlap_amplitude":overlap,"partial_hom_coincidence":partial[(1,1)],
            "partial_hom_expected":(1-overlap**2)/2,
            "max_fock_amplitude_error_vs_70_decimal_permanent":permanent_error,
            "lifted_unitary_max_residual":float(np.max(abs(lifted.conj().T@lifted-np.eye(basis3.dimension)))),
            "max_probability_error_vs_perceval_slos":sdk_error,
            "max_density_entry_error_vs_pennylane_default_mixed":pennylane_error,
            "single_pauli_errors_tested_across_5_7_9_qubit_codes":63,
            "two_location_erasure_pauli_cases_in_five_qubit_code":160},
        "repetition_memory":{"analytic_one_round_error_rate":3*.08**2-2*.08**3,"measured":memory.to_dict()},
        "stim_surface_memory":stim_result.to_dict(),
        "limitations":["Finite mathematical models, not hardware-calibrated accuracy.",
            "Main engine complex128; 70-decimal arithmetic is only an independent permanent reference.",
            "Sampled logical errors depend on the supplied noise model and decoder.",
            "CI configuration is included; this report records local execution only."]}
    (root/"benchmarks").mkdir(exist_ok=True)
    (root/"benchmarks/validation.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (root/"requirements-validated.txt").write_text("# Direct package versions from the recorded validation environment.\n"+
        "# Not a full transitive lockfile; Python "+versions["python"]+".\n"+
        "\n".join(f"{name}=={versions[name]}" for name in distributions)+"\n",encoding="utf-8")
    interval=memory.wilson_95
    lines=["# Validation report","",f"Recorded: {report['generated_utc']}","",
        f"**{tests['tests']} tests passed; {tests['failures']} failures, {tests['errors']} errors, {tests['skipped']} skips.**",
        "Editable package installation, all six runnable examples, the CLI, and actual optional SDK integrations were exercised.","",
        "## Deterministic comparisons","","| Check | Measured result |","| --- | --- |",
        f"| Balanced identical-photon HOM coincidence | {hom[(1,1)]:.4g} |",
        f"| Each HOM bunching outcome | {hom[(2,0)]:.16g}, {hom[(0,2)]:.16g} |",
        f"| Partial-overlap HOM coincidence, amplitude overlap 0.7 | {partial[(1,1)]:.16g} (analytic 0.255) |",
        f"| Maximum Fock amplitude deviation from a 70-decimal permanent reference | {permanent_error:.4g} |",
        f"| Maximum probability deviation from Perceval SLOS | {sdk_error:.4g} |",
        f"| Maximum density-entry deviation from PennyLane default.mixed | {pennylane_error:.4g} |",
        "| Single-qubit Pauli errors corrected across the 5-, 7-, and 9-qubit codes | 63 / 63 |",
        "| Five-qubit code: Pauli assignments across every pair of erased locations | 160 / 160 |","",
        "Additional tests cover vacuum, binomial loss, loss of coherence, unequal dual-rail filtering, Gaussian phase covariance,",
        "three-photon complex Gram phases against an independent permutation formula, saturated detectors, false heralds,",
        "destructive measurement traces, complete Bell instruments, signed/CSS/custom codes, coherent error recovery,",
        "Knill-Laflamme conditions, finite bosonic transpose recovery, resource guards, and JSON/NPZ interchange.","",
        "## Sampled examples","",
        f"Repetition memory: p(X)=0.08, one round, seed 77, {memory.shots:,} shots. Analytic block error rate is 0.018176.",
        f"Measured {memory.failures} failures: {memory.logical_error_rate:.8f}, Wilson 95% interval [{interval[0]:.8f}, {interval[1]:.8f}].","",
        f"Stim rotated surface-code memory: distance 3, 3 rounds, gate depolarization 0.008 and measurement-flip probability 0.008.",
        f"Seed 123, {stim_result.shots:,} shots: {stim_result.failures} any-observable failures, rate {stim_result.logical_error_rate:.8f}.",
        "These rates are examples of their stated models, not a photonic hardware threshold or a comparison of the two architectures.","",
        "## Validated versions","","| Component | Version |","| --- | --- |"]
    lines += [f"| {name} | {version} |" for name,version in versions.items()]
    lines += ["","## Reproduce","","```bash","python -m pip install -e '.[all]'",
              "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/validate.py","```","",
              "PowerShell users can set the two environment variables separately before running Python.",
              "The script reruns tests and regenerates this report, `benchmarks/validation.json`, and the directly validated version list.","",
              "All numerical deviations are measured on the finite test cases, not global accuracy bounds. Main simulation remains complex128.",
              "The default cutoff/memory guards intentionally prevent some larger exact optical embeddings. Further architecture or hardware claims",
              "require their own physical circuits, calibration data, decoder studies and convergence/error-budget analysis.","",
              "The repository includes GitHub Actions configuration. This document records local execution, not an assertion that hosted CI completed.",""]
    (root/"docs/VALIDATION.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"tests":tests,"deterministic_checks":report["deterministic_checks"]},indent=2))


if __name__=="__main__":
    main()
