# PLQ — Photonic Logical Qubits

A Python simulator for **photonic circuits, logical qubits and quantum error correction**. Keep photon loss, detector errors, heralding probabilities and numerical cutoffs visible.

**v0.4.0** · Python 3.11+ · NumPy + SciPy · [Full guide](docs/GUIDE.md) · [Paper benchmarks](docs/PAPER_BENCHMARKS.md)

## New in 0.4: scale optics and make hardware assumptions testable

- **Sparse pure-state optics:** `Circuit.run_sparse` avoids the full Fock density
  matrix and lifted unitary. `Circuit.sample` adds explicit loss/phase trajectories
  with seeds, shots and confidence intervals. [Scaling guide](docs/SCALABILITY.md).
- **Separate evidence categories:** paper-parameter runs are tagged separately
  from experimental-data comparisons. `compare-experiment` checks raw-count
  denominators, dataset hashes and measurement conventions. [Reproduction contract](docs/REPRODUCTION.md).
- **A physical optical-to-QEC component:** six-rail teleportation preserves detector
  events, false heralds, surviving output states, buffer loss and feedforward
  deadlines. [Hardware bridge and remaining gaps](docs/HARDWARE_BRIDGE.md).

```bash
python examples/scalable_optics.py
python -m plq teleportation examples/configs/teleportation.json
python -m plq compare-experiment examples/experiments/synthetic_manifest.json
```

The comparison example uses synthetic counts. None of these commands claims to
reproduce an entire experiment or demonstrate a fault-tolerant hardware threshold.

## Install and run

```bash
git clone https://github.com/ht13255/PLQ.git
cd PLQ
python -m venv .venv
```

Activate the environment:

| System | Command |
| --- | --- |
| macOS / Linux | `source .venv/bin/activate` |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows cmd | `.venv\Scripts\activate.bat` |

```bash
python -m pip install -e .
python -m plq hom --overlap 0.7 --transmission 0.9
```

The coincidence probability should be approximately **0.20655**. `plq` and `python -m plq` run the same CLI. `python -m plq doctor` shows installed capabilities. Install from this repository; a PyPI release is not assumed.

## A circuit in four lines

```python
from plq import Circuit

result = Circuit(2).bs(0, 1).loss(0, 0.9).loss(1, 0.9).run([1, 1])
print(result.probabilities())
print(result.state.diagnostics())
```

`[1, 1]` means one photon in each input mode. The initial total-photon cutoff is inferred. Vacuum, photon loss and bunching remain in the state. `transmission=0.9` is a **photon-survival probability**, not a field amplitude. For mixed states and coherent superpositions, use [`FockState`](docs/GUIDE.md#start-with-an-optical-circuit).

## Choose the calculation you need

| Goal | Run / API |
| --- | --- |
| Try published source parameters | `python -m plq source-hom examples/papers/somaschi_2016.json` |
| Reproduce paper comparisons and sensitivity runs | `python scripts/paper_benchmarks.py` |
| Interfere independent mixed spectral states | `python examples/mixed_wavepackets.py` |
| Lossy optics with detectors and heralding | `python -m plq optics examples/configs/optics.json` |
| Thermal bath with a reported truncation error | `python examples/thermal_accuracy.py` |
| Exact small-code Pauli error rate | `python examples/exact_memory.py` |
| Repeated noisy memory, erasure flags and confidence intervals | `python -m plq memory examples/configs/memory.json` |
| Connect optical rails to a logical code | `python examples/optical_logical.py` |
| Custom finite bosonic encoding and recovery | `python examples/custom_bosonic.py` |
| Large Clifford syndrome-extraction circuits | `python examples/surface_code_stim.py` (optional SDKs) |

Add `--output results/run.json` to `optics` or `memory` to save the configuration, model, versions and results. Optical JSON now infers the initial cutoff when omitted. A custom code file in a memory JSON is resolved relative to that JSON file.

## Use measured source parameters

```python
from plq import number_distribution_from_moments, source_hom

# Somaschi et al. source-summary parameters; assume no n>=3 component.
p = number_distribution_from_moments(mean_photons=0.154, g2_zero=0.0028)
result = source_hom([p, p], indistinguishability=0.9956,
                    multiphoton_model="same_wavepacket")
print(result["parallel"]["coincidence_probability"])  # 8.279879649e-5
print(result["click_visibility"])  # 0.9930324768
```

`mean_photons` is the mean photon number, and `g2_zero` is a normalized factorial moment. Neither is simply the probability of two photons. The helper explicitly assumes support **n=0,1,2**, rejects inconsistent moments, and does not infer a unique physical source from two measurements.

Extra photons must have a declared model: `same_wavepacket`, or `orthogonal_noise` (one signal plus one source-specific orthogonal noise photon in the two-photon sector). The latter example gives visibility about **0.9928183329**. These are alternative assumptions, not confidence bounds. Source vacuum, multiphoton events, propagation loss, detector efficiency and dark counts all remain in the calculation. [Conventions and limits](docs/PHYSICS.md#imperfect-number-sources-and-hom-observables).

**Overlap units matter.** `plq hom --overlap 0.7` uses an amplitude. `plq hom --indistinguishability 0.985` uses its square and gives ideal coincidence probability 0.0075. Raw measured HOM visibility also includes the source, detector and normalization convention; it cannot generally be substituted for intrinsic overlap.

The [paper report](docs/PAPER_BENCHMARKS.md) records actual runs using Somaschi, Ding and Menssen parameters, independent formula comparisons, and one-parameter sensitivity runs. It distinguishes source-summary scenarios from experimental reproduction. The supplied pre-etalon Ding efficiency budget predicts 3.658 million counts/s versus the reported 3.7 million; the rounded factors are not fitted.

## Mixed spectra and measured lossy circuits

```python
import numpy as np
from plq import Circuit, mixed_wavepacket_input

rho = np.diag([0.7, 0.3])  # one photon's density matrix in a common internal basis
photons = mixed_wavepacket_input(2, [0, 1], [rho, rho])
print(photons.through(Circuit(2).bs(0, 1)).spatial_probabilities()[(1, 1)])
# 0.21, from (1 - Tr(rho @ rho))/2

# A is a field-amplitude transfer matrix, including coherent mixing and loss.
A = np.array([[0.45 + 0.1j, 0.2], [-0.15j, 0.57 - 0.04j]])
result = Circuit(2).transfer(A).run([1, 1])
print(result.state.trace)  # approximately 1; lost-photon sectors are retained
print(result.transfer_residuals)
```

Mixed inputs require independent photons in distinct input ports and matrices in the same orthonormal internal basis. Three-photon results retain `Tr(rho1 @ rho2 @ rho3)`, which pairwise HOM overlaps cannot supply. `Circuit.transfer` accepts square passive contractions and implements vacuum loss using a singular-value decomposition. It rejects gain and reports reconstruction residuals. It does not infer a calibration from intensities alone.

## Accuracy controls that change the calculation

**Thermal attenuation.** A finite-temperature bath can add photons as well as absorb them. PLQ evolves a beam-splitter interaction with an explicit thermal environment, and enlarges the system basis to retain the added photons:

```python
from plq import Circuit

result = Circuit(1).thermal_loss(
    0, transmission=0.7, mean_photons=0.2, tail_tolerance=1e-13
).run([0])
print(result.bath_omitted_probability)
print(result.numerical_trace_error)
```

The bath tail is **not renormalized away**. `tail_tolerance` bounds each bath's omitted probability, not experimental accuracy or floating-point error. `bath_omitted_probability` reports accumulated omitted weight; rare-event conditional results need a much smaller tail than their herald probability. Explicit `bath_cutoff` and cutoff-convergence examples are supported. Thermal noise is not detector dark counts. [Model and equations](docs/PHYSICS.md#thermal-attenuation-and-controlled-truncation).

**Exact error rates.** For a small code with independent Pauli errors and one ideal syndrome/recovery round, enumerate all supported errors instead of estimating a rare failure rate from shots:

```python
from plq import five_qubit_code, MemoryNoise, exact_pauli_memory

result = exact_pauli_memory(
    five_qubit_code(), MemoryNoise(px=0.001, py=0.001, pz=0.15)
)
print(result.logical_error_rate)  # approximately 0.03643846138
```

The default decoder sums probabilities of errors with the same logical effect before choosing a correction. For this specified biased-noise example, minimum-weight decoding gives approximately 0.16810259224. These are **two decoders under the same ideal one-round model**, not hardware error rates or thresholds. Nonzero erasure/readout noise is rejected by this exact-rate API; use memory trials or Stim for those experiments. [Decoder contract](docs/PHYSICS.md#exact-pauli-enumeration-and-degenerate-maximum-likelihood).

**Resource limits are explicit.** Exact Fock density matrices and exhaustive decoders can grow exponentially. There is no silent approximation when a budget is exceeded:

```python
from plq import Circuit, Precision

precision = Precision(atol=1e-12, max_dimension=4096,
                      max_kraus_bytes=2 * 1024**3)
result = Circuit(2).bs(0, 1).run([2, 2], precision=precision)
```

A dense complex128 matrix needs `16 * dimension**2` bytes, and operations need several arrays. Larger limits permit more computation; they do not change floating-point precision. The separate mpmath permanent reference checks selected amplitudes with adjustable decimal precision.

## Optional integrations

Install only what you use:

| Integration | Install | Example |
| --- | --- | --- |
| Quandela Perceval | `python -m pip install -e ".[perceval]"` | `python examples/perceval_bridge.py` |
| PennyLane | `python -m pip install -e ".[pennylane]"` | `python examples/pennylane_bridge.py` |
| Stim + PyMatching | `python -m pip install -e ".[qec]"` | `python examples/surface_code_stim.py` |
| Tests + high-precision reference | `python -m pip install -e ".[test,reference]"` | `python scripts/validate.py --core` |
| Everything | `python -m pip install -e ".[all]"` | `python scripts/validate.py` |

Perceval exchanges numeric lossless optical unitaries. PennyLane uses explicit Kraus channels and success/failure flags. Stim uses the supplied Clifford circuit and detector error model. These bridges do not automatically synthesize fault-tolerant photonic hardware. Pasqal/Pulser is a neutral-atom SDK; the optical bridge is Quandela Perceval.

## Validation and scope

The 0.4 changes have a separate [scaling and hardware validation record](docs/VALIDATION_V04.md).

v0.3 preserves small positive Gram eigenvalues and uses a stable two-wavepacket factorization: a tested coincidence near 1e-14 no longer vanishes at the default tolerance. New tests check mixed-state density invariants, multiphoton source moments, and lossy transfer maps against an independent vacuum dilation. Earlier thermal, detector, decoder and SDK tests remain. See the [measured validation report](docs/VALIDATION.md), [paper comparisons](docs/PAPER_BENCHMARKS.md) and [GitHub CI](https://github.com/ht13255/PLQ/actions).

PLQ is a **finite-model research simulator**, not a hardware-calibrated digital twin. Main calculations use complex128. Defaults are ideal until noise is specified. Ideal encoding, logical gates and recovery are not physical fault-tolerant schedules. Active squeezing, nonlinear optics, detector dead time/afterpulsing and correlated mixed spectral inputs still need additional models. `WavepacketState.through` does not accept thermal baths: a spectral bath population must be specified explicitly.

- [Scaling, sparse states and sampling](docs/SCALABILITY.md)
- [Experimental comparison contract](docs/REPRODUCTION.md)
- [Optical hardware instrument](docs/HARDWARE_BRIDGE.md)
- [Paper inputs, measured runs and limitations](docs/PAPER_BENCHMARKS.md)
- [Full examples and SDK guide](docs/GUIDE.md)
- [Physics, conventions and accuracy limits](docs/PHYSICS.md)
- [API and JSON formats](docs/API.md)
- [Custom codes, decoders and backend compatibility](docs/EXTENDING.md)
- [Primary references](docs/REFERENCES.md)
