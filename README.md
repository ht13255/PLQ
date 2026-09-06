# PLQ — Photonic Logical Qubits

PLQ is a Python reference simulator connecting **photonic circuits, encoded logical states, quantum error correction, and quantum algorithms**. Its short API keeps physical assumptions visible: photon loss remains loss, heralding keeps its success probability, and an ideal encoded gate is explicitly distinguished from a synthesized optical gate.

**Status: v0.1.0, a tested finite-model research implementation.** It is not a hardware-calibrated digital twin, a complete photonic fault-tolerance compiler, or a promise that every existing QEC software package runs unchanged. See [the compatibility contract](docs/EXTENDING.md) and [physical assumptions](docs/PHYSICS.md).

The optical SDK bridge is **Quandela Perceval**. **Pasqal/Pulser uses neutral atoms** and is not a photonic circuit SDK; PLQ does not silently reinterpret its analog Hamiltonians as optics. See the [Perceval documentation](https://perceval.quandela.net/docs/v1.2/index.html) and [Pasqal documentation](https://docs.pasqal.com/pulser/programming/).

## Install

Python 3.11 or newer. NumPy and SciPy are the only required runtime dependencies. Use a virtual environment:

```bash
git clone https://github.com/ht13255/PLQ.git
cd PLQ
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[all]'
python -m pytest -q
```

On Windows activate with `.venv\Scripts\activate`. For the core only, run `python -m pip install -e .`. Optional groups are `perceval`, `pennylane`, `qec` (Stim and PyMatching), `reference` (mpmath), and `test`.

The distribution name is `plq-sim`; the import is `plq`. Install this repository: a PyPI release is not assumed. [requirements-validated.txt](requirements-validated.txt) records the versions directly used for validation; it is not a complete transitive lockfile.

## Start with an optical circuit

```python
from plq import FockBasis, FockState, Circuit

basis = FockBasis(modes=2, max_photons=2)
input_state = FockState.ket(basis, (1, 1))

circuit = (
    Circuit(2)
    .bs(0, 1, transmission=0.5)
    .loss(0, transmission=0.9)
    .loss(1, transmission=0.9)
)
result = circuit.run(input_state)
print(result.probabilities())
print(result.state.diagnostics())
```

Here `transmission` is a **power/photon-survival probability**, not an amplitude. The full state includes vacuum and lower photon-number sectors; its trace stays one. The balanced beam splitter produces Hong–Ou–Mandel bunching before the losses.

`max_photons` is a **total photon-number cutoff across all modes**. Passive optics preserves photon number and pure loss only decreases it, so this basis does not discard bunching. Mode occupations may reach the full total cutoff.

## Simulate an encoded logical qubit

```python
import numpy as np
from plq import LogicalQPU, five_qubit_code, PAULI

psi = np.array([np.sqrt(0.3), 1j * np.sqrt(0.7)])
qpu = (
    LogicalQPU(five_qubit_code())
    .prepare(psi)
    .physical_gate(PAULI["Y"], wires=[2])
    .correct()
)
logical = qpu.logical_state()
print(logical.probability)             # weight in the logical code space
print(logical.conditional())          # normalization is an explicit choice
```

This example uses an **ideal projective stabilizer recovery map**. It does not claim a physical circuit with error-free ancillas or deterministic optical entangling gates.

Built-in code factories:

| Factory | Code / protection |
| --- | --- |
| `repetition_code(n, basis="Z")` | Bit-flip repetition code; does not correct arbitrary Pauli errors |
| `repetition_code(n, basis="X")` | Phase-flip repetition code |
| `five_qubit_code()` | Five-qubit `[[5,1,3]]` stabilizer code |
| `steane_code()` | Steane `[[7,1,3]]` CSS code |
| `shor_code()` | Shor `[[9,1,3]]` code |
| `StabilizerCode(...)` | User-defined independent signed commuting generators; any number of logical qubits within resources |
| `css_code(hx, hz, ...)` | User-defined binary CSS check matrices, including sparse matrix inputs |
| `CodeSpace(V)` | Any finite-dimensional orthonormal encoding isometry, including non-stabilizer and bosonic code spaces |

## Connect real optical evolution to QEC

[`examples/optical_logical.py`](examples/optical_logical.py) performs a complete small-block connection:

1. Encode a logical state in three physical qubits.
2. Embed those qubits into six optical dual-rail modes.
3. Apply an actual optical rail swap and per-mode photon loss.
4. Extract the unnormalized computational branch.
5. Apply ideal QEC recovery and report both survival probability and conditional fidelity.

`DualRail` uses `|0> = |1,0>`, `|1> = |0,1>`. Qubit zero is the most significant tensor factor. `DualRail.decode()` is an ideal computational-subspace projection, not a physical nondestructive photon counter. Leakage, vacuum and bunching remain outside its retained branch.

For equal transmission `eta` on both rails, the qubit survival probability is `eta`. Unequal rail loss also filters the logical amplitudes; PLQ preserves that state dependence.

## Detailed physical controls

| Control | API | Meaning / unit |
| --- | --- | --- |
| Beam splitter | `Circuit.bs(..., transmission=T, phase=phi)` | Power transmission `T` in `[0,1]`, relative phase in radians |
| Arbitrary passive interferometer | `Circuit.unitary(U, modes=...)` | Explicit finite unitary on ordered modes |
| Optical phase | `Circuit.phase(mode, radians)` | Phase in radians |
| Propagation/component loss | `Circuit.loss(mode, transmission=eta)` | Vacuum-environment pure loss, independently located in circuit order |
| Correlated phase fluctuations | `Circuit.phase_noise(covariance, mean)` | Covariance in radians squared; mean in radians |
| Source contamination | `independent_sources(basis, distributions)` | Explicit per-mode vacuum, single-photon and multiphoton probabilities |
| Partial distinguishability | `wavepacket_input(..., gram=G)` | Full complex positive-semidefinite wavepacket Gram matrix |
| Time/frequency mismatch | `gaussian_gram(times, angular_frequencies, sigma)` | Seconds, radians/second, intensity temporal standard deviation in seconds |
| Detector quantum efficiency | `Detector(efficiency=...)` | Conditional photon registration probability |
| Dark counts | `dark_rate_hz`, `gate_seconds` | Poisson rate in counts/second and acquisition window in seconds |
| Arrival-window acceptance | `arrival_offset_seconds`, `timing_sigma_seconds` | Independent Gaussian arrival-time acceptance model |
| Detector type | `threshold=True` or `saturation=k` | Click/no-click, or photon number resolving; the last PNR bin means `>= k` |
| Measurement conditioning | `herald(state, modes, counts, detectors)` | Destructive number-diagonal POVM; returns absolute branch probability |
| Numerical/resource controls | `Precision(...)` | Absolute tolerance, Hilbert dimension, Kraus count and storage budgets |
| Phenomenological memory | `MemoryNoise(...)` | Per-round Pauli, erasure/replacement and syndrome-readout probabilities |

Defaults are ideal except for the explicit finite numerical precision and resource limits. They are **not measured parameters of a particular vendor's QPU**. State the parameter calibration and uncertainty for any hardware comparison.

### Distinguishable photons

```python
from plq import Circuit, wavepacket_input

overlap = 0.7  # wavepacket amplitude overlap, not HOM visibility
photons = wavepacket_input(2, [0, 1], [[1, overlap], [overlap, 1]])
out = photons.through(Circuit(2).bs(0, 1))
print(out.spatial_probabilities()[(1, 1)])  # (1 - overlap**2) / 2 = 0.255
```

The Gram matrix is factored into explicit orthogonal internal modes. Internal modes are retained through evolution and summed only when obtaining spatial count probabilities. Complex overlap phases are retained, including genuinely multiphoton effects. This constructor describes products of pure wavepackets, not arbitrary correlated mixed spectral states.

### Detectors and heralding

```python
from plq import Detector, herald

detector = Detector(efficiency=0.92, dark_rate_hz=100,
                    gate_seconds=2e-9, saturation=3)
branch = herald(result.state, modes=[0], counts=[1], detectors=[detector])
print(branch.probability)
if branch.probability > 0:
    remaining = branch.conditional_state()
```

The unnormalized remaining state is also available as `branch.remaining_state`. If all modes were measured it is `None`. Finite detector efficiency and dark counts can cause false heralds. The four-mode Bell analyzer reports the physical 50% ideal average success of an unboosted analyzer, with separate `psi_plus`, `psi_minus` and `failure` outcomes; see [`fusion.py`](src/plq/fusion.py).

## Perceval bridge

```python
import perceval as pcvl
from plq.adapters.perceval import from_perceval, to_perceval

design = pcvl.Circuit(2).add(0, pcvl.BS.H())
plq_design = from_perceval(design)
roundtrip = to_perceval(plq_design)
plq_design.loss(0, transmission=0.95)
```

The bridge exchanges the **numeric lossless single-particle unitary**, preserving its exact phase convention. Importing a `Processor` is rejected because its sources, noise, heralds and postselection cannot be inferred from a circuit unitary. Exporting a noisy PLQ circuit as a lossless unitary is also rejected. Explicit polarization modes must be expanded by the user. [Perceval's component reference](https://perceval.quandela.net/docs/v1.2/reference/components/unitary_components.html) describes its optical conventions.

## PennyLane bridge

Use a PLQ optical success map as a **trace-preserving flagged channel** in a PennyLane algorithm:

```python
import pennylane as qml
from plq import Circuit, DualRail
from plq.adapters.pennylane import to_operation

success = DualRail(1).effective_channel(
    Circuit(2).loss(0, 0.7).loss(1, 0.9)
)
channel = success.flagged()
device = qml.device("default.mixed", wires=["flag", "data"])

@qml.qnode(device)
def algorithm(theta):
    qml.RY(theta, wires="data")
    to_operation(channel, ["flag", "data"])
    return qml.probs(wires=["flag", "data"])

print(algorithm(0.6))
```

The first wire is the flag: input is flag zero, success remains in the data block, and failure occupies index `d` for a `d`-dimensional data register. Padding states are invariant. A trace-decreasing map cannot be passed directly to `QubitChannel`; PLQ rejects that mistake. This uses PennyLane's [fixed Kraus-channel API](https://docs.pennylane.ai/en/stable/code/api/pennylane.QubitChannel.html).

The reverse bridge, `from_operations(operations, wire_order=[...])`, imports numeric finite-qubit unitary and channel operations. State preparation, continuous-variable operations and measurements are rejected. `logical_unitary(qfunc, *args, wire_order=...)` imports an operation-only algorithm for `LogicalQPU.logical_gate(U)`. Its encoded gate is ideal and does not synthesize an optical circuit.

PennyLane can differentiate **algorithm parameters around a fixed imported channel**. Use `@qml.qnode(device, diff_method="parameter-shift")` for the validated gradient route: in the tested SDK version, default backpropagation through this rank-deficient flagged channel can return NaN. PLQ's NumPy circuit simulation and conversion are not an autodifferentiable optical backend. [`examples/pennylane_bridge.py`](examples/pennylane_bridge.py) checks the actual gradient against an analytic derivative.

## Larger QEC experiments

```python
from plq import five_qubit_code, MemoryNoise, simulate_memory

result = simulate_memory(
    five_qubit_code(),
    MemoryNoise(px=0.003, py=0.001, pz=0.008, erasure=0.01),
    shots=10000, rounds=3, seed=2026,
)
print(result.to_dict())
```

The memory simulator propagates Pauli frames without dense wavefunctions. Error probabilities can be scalar or per-qubit lists; `syndrome_flip` can be per-check. The default decoder is bounded reference minimum-weight lookup, with an erasure-aware variant when flags are present. A custom decoder can use the entire measurement history.

**Memory erasure has a specific abstraction:** the lost physical qubit is ideally replenished by a maximally mixed replacement, its location is reported, and its unknown Pauli frame is sampled uniformly. This is not the same object as a vacuum Fock mode. Every non-stabilizer residual counts as a block failure, including any logical Pauli; decoder failures are counted and included conservatively. The optional final perfect round is explicit. See [PHYSICS.md](docs/PHYSICS.md).

For large Clifford circuits with explicit noisy syndrome extraction, use your own **Stim circuit and decoder**:

```python
import stim
from plq.adapters.stim import simulate_stim

circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z", distance=5, rounds=5,
    after_clifford_depolarization=0.003,
    before_measure_flip_probability=0.003,
)
result = simulate_stim(circuit, shots=10000, seed=2026)
# Or: circuit = stim.Circuit.from_file("my_code.stim")
```

The default uses PyMatching. A supplied `decode_batch(detections)` implementation may replace it. This path follows [Stim's supported stabilizer circuit model](https://github.com/quantumlib/Stim) and [PyMatching's detector graph assumptions](https://pymatching.readthedocs.io/en/stable/). It does not turn coherent optics into Pauli noise automatically or silently discard unsupported detector error terms.

## Your own QEC code

```python
from plq import StabilizerCode, LogicalQPU

my_code = StabilizerCode(
    generators=["ZZI", "IZZ"],
    logical_x=["XXX"], logical_z=["ZII"],
    name="My bit-flip code",
)
my_code.save("my_code.json")
qpu = LogicalQPU(my_code)
```

For an arbitrary code use `CodeSpace(V)`, then supply a physical `KrausChannel` and a recovery channel. `knill_laflamme(code, errors)` tests the specified error set. `transpose_recovery(code, noise)` constructs a general approximate recovery, with a documented pseudoinverse cutoff and CPTP completion. A working finite binomial photon-loss example is included in [`examples/custom_bosonic.py`](examples/custom_bosonic.py).

`CodeSpace.save/load` uses a non-pickle NPZ file. `register_code(name, factory)` adds your own Python factory. JSON configuration never executes code. Subsystem, Floquet, fusion, LDPC, cat and GKP workflows have different schedules, measurements and decoders; see [EXTENDING.md](docs/EXTENDING.md) for the exact supported connection points.

## Configuration files and reproducible runs

```bash
plq hom --overlap 0.7 --transmission 0.9
plq optics examples/configs/optics.json --output results/optics.json
plq memory examples/configs/memory.json --output results/memory.json
python examples/optical_logical.py
python examples/custom_bosonic.py
python examples/perceval_bridge.py
python examples/pennylane_bridge.py
python examples/surface_code_stim.py
```

`python -m plq.cli ...` is equivalent. Outputs retain the input configuration, absolute probabilities, selected model, RNG seed when sampling, and version metadata. Unknown configuration fields fail explicitly. Optical probability results are deterministic density-matrix calculations; memory and Stim results include 95% Wilson binomial confidence intervals. Zero sampled failures does not imply zero underlying failure probability.

## Accuracy, validation and resource limits

PLQ uses `complex128`. `Precision(atol=1e-10)` is an **absolute validation tolerance**, not a promise of `1e-10` experimental accuracy or a bound on every accumulated numerical error. Input states, isometries, Kraus completeness, covariance matrices and unitaries are validated. States are never silently made pure, normalized, or projected positive. Negligible negative probability entries caused by floating-point roundoff may be clipped when exposing Fock probability dictionaries; the stored matrix and diagnostics remain unchanged.

`plq.reference` provides a separate mpmath permanent calculation with adjustable decimal precision. The main simulator remains `complex128`; high-precision arithmetic cannot restore digits absent from its input matrices.

The complete Fock space dimension is `comb(modes + max_photons, max_photons)`. Defaults are a 1,024-dimensional Hilbert space, at most 4,096 explicit Kraus matrices and at most 512 MiB of Kraus storage. Operations need additional arrays. Pure-loss state evolution uses sparse index updates; explicit Kraus extraction is intended for small instruments. Dense QEC recovery can exceed the Kraus memory limit before the dimension limit. Limits raise `ResourceLimitError` and can be configured deliberately.

For one photon per dual-rail qubit, three physical qubits use 84 basis states, four use 495, and five use 3,003. The five-qubit optical embedding therefore exceeds the default dense dimension budget. Use the stabilizer or Stim path for large codes, and the optical path to characterize small physical components. There is no claim of efficient exact full-Fock simulation of an industrial QPU.

Validation covers analytic interference/loss/phase formulas, complex multiphoton distinguishability, destructive heralding, Bell measurement, arbitrary-code recovery, all single-qubit Pauli errors of the built-in distance-three codes, and real SDK cross-checks. See the measured [validation report](docs/VALIDATION.md) and [machine-readable results](benchmarks/validation.json).

## Scope that must remain explicit

- Passive interferometers, vacuum-environment loss and the specified Gaussian phase model are implemented. Active squeezing, thermal environments, nonlinear optics, homodyne detection, detector dead time/afterpulsing and time-correlated drift require additional models.
- Ideal code encoding, ideal logical gates and ideal recovery are not fault-tolerant hardware schedules. General linear-optical gate synthesis, automatic resource-state construction and universal feed-forward compilation are not implemented.
- User-supplied finite code spaces are supported. Complete finite-squeezing GKP/cat decoders or all third-party QEC packages are not built in; import an appropriate representation and supply the missing instrument/decoder.
- Monte Carlo confidence intervals quantify sampling uncertainty within the chosen model. They do not quantify uncertain calibration parameters, model mismatch or extrapolation to another architecture.

## Documentation

- [Physical models, equations and conventions](docs/PHYSICS.md)
- [Code/decoder/backend compatibility and extension guide](docs/EXTENDING.md)
- [JSON configuration and public API guide](docs/API.md)
- [Measured validation report](docs/VALIDATION.md)
- [Primary references and SDK sources](docs/REFERENCES.md)
