# PLQ — Photonic Logical Qubits

A Python simulator for **photonic circuits, logical qubits and quantum error correction**. Keep photon loss, detector errors, heralding probabilities and numerical cutoffs visible.

**v0.2.0** · Python 3.11+ · NumPy + SciPy · [한국어 사용법](docs/QUICKSTART.ko.md) · [Full guide](docs/GUIDE.md)

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
| Lossy optics with detectors and heralding | `python -m plq optics examples/configs/optics.json` |
| Thermal bath with a reported truncation error | `python examples/thermal_accuracy.py` |
| Exact small-code Pauli error rate | `python examples/exact_memory.py` |
| Repeated noisy memory, erasure flags and confidence intervals | `python -m plq memory examples/configs/memory.json` |
| Connect optical rails to a logical code | `python examples/optical_logical.py` |
| Custom finite bosonic encoding and recovery | `python examples/custom_bosonic.py` |
| Large Clifford syndrome-extraction circuits | `python examples/surface_code_stim.py` (optional SDKs) |

Add `--output results/run.json` to `optics` or `memory` to save the configuration, model, versions and results. Optical JSON now infers the initial cutoff when omitted. A custom code file in a memory JSON is resolved relative to that JSON file.

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

v0.2 fixes rare detector probabilities rounding to zero, large-photon-count overflow, and ambiguous configuration inputs. Regression tests independently compare thermal evolution with a full system-plus-environment Fock calculation, check thermal tails and limiting cases, and compare the new decoder with explicit stabilizer-group probability sums. See the [measured validation report](docs/VALIDATION.md) and [GitHub CI](https://github.com/ht13255/PLQ/actions).

PLQ is a **finite-model research simulator**, not a hardware-calibrated digital twin. Main calculations use complex128. Defaults are ideal until noise is specified. Ideal encoding, logical gates and recovery are not physical fault-tolerant schedules. Active squeezing, nonlinear optics, detector dead time/afterpulsing and correlated mixed spectral inputs still need additional models. `WavepacketState.through` does not accept thermal baths: a spectral bath population must be specified explicitly.

- [한국어 시작 안내](docs/QUICKSTART.ko.md)
- [Full examples and SDK guide](docs/GUIDE.md)
- [Physics, conventions and accuracy limits](docs/PHYSICS.md)
- [API and JSON formats](docs/API.md)
- [Custom codes, decoders and backend compatibility](docs/EXTENDING.md)
- [Primary references](docs/REFERENCES.md)
