# Public API and configuration

## Core objects

| Object / function | Principal operations |
| --- | --- |
| `Precision` | `atol=1e-10`, `max_dimension=1024`, `max_kraus=4096`, `max_kraus_bytes=512*2**20` |
| `FockBasis(modes, max_photons, precision=...)` | `states`, `index`, `occupations`, `dimension` |
| `FockState` | `ket`, `amplitudes`, `mixture`, `probabilities`, `diagnostics`, `postselect`, `conditional` |
| `Circuit(modes)` | Fluent `bs`, `phase`, `unitary`, `loss`, `phase_noise`; `run(state)`, `channel(basis)` |
| `DualRail(n_qubits, basis=..., precision=...)` | `encode`, `decode`, `effective_channel`, `isometry` |
| `Detector` | `response(photons)`, `effective_efficiency`, `outcomes` |
| `herald` | Absolute branch probability, unnormalized remaining state, measured/remaining mode order |
| `wavepacket_input` | Explicit internal-mode state, `through`, `spatial_probabilities`, `detection_probabilities` |
| `KrausChannel` | Validated operators; `apply`, `then`, `unitary`, `flagged` |
| `CodeSpace` | `encode`, `decode`, `save`, `load`; arbitrary finite isometry |
| `StabilizerCode` | `codespace`, `syndrome`, `syndrome_projector`, `recovery_channel`, JSON persistence |
| `LogicalQPU` | `prepare`, `prepare_physical`, `physical_gate`, `logical_gate`, `local_channel`, `channel`, `correct`, `logical_state`, `fock_state` |
| `simulate_memory` | Pauli-frame block trials with `MemoryNoise`, shots, rounds, decoder, seed, final perfect round |
| `simulate_stim` | User circuit, optional batch decoder, shots, seed, correlated matching |
| `knill_laflamme` | Correctability residual for a supplied error set |
| `transpose_recovery` | General approximate recovery for a finite square CPTP noise channel |

The Python docstrings describe method-level shapes and exceptions. Arrays use `complex128`. For a pure state pass a vector; for a mixed state pass a square density matrix. Probability normalization is validated, and subnormalization must be explicit where indicated. `Branch.conditional()` returns a normalized density matrix; `FockState.conditional()` returns a FockState.

## Optical JSON format

Run `plq optics examples/configs/optics.json`. Allowed top-level fields:

| Field | Meaning |
| --- | --- |
| `modes` | Required positive integer |
| `max_photons` | Required nonnegative total cutoff |
| `precision` | Optional `Precision` constructor fields |
| `input` | Exactly one of the input forms below |
| `steps` | Ordered optical operations |
| `detectors` | Optional one detector configuration per spatial mode |
| `herald` | Optional `modes`, `counts`, and per-measured-mode `detectors` |
| `dualrail_qubits` | Optional ideal final computational-subspace projection; requires exactly twice as many modes |

Input forms:

```json
{"occupation": [1, 1]}
```

```json
{"sources": [[0.1, 0.8, 0.1], [0.2, 0.8]]}
```

```json
{"amplitudes": [
  {"occupation": [1, 0], "real": 0.6, "imag": 0.0},
  {"occupation": [0, 1], "real": 0.0, "imag": 0.8}
]}
```

Source lists index photon number starting at vacuum. Each source distribution must sum to one. An omitted total-cutoff tail is reported without normalization. Amplitude inputs must already be normalized, and duplicate occupations are rejected.

Each step has an `operation` field and the same named arguments as the Python method:

```json
[
  {"operation": "bs", "first": 0, "second": 1, "transmission": 0.5, "phase": 0.2},
  {"operation": "phase", "mode": 0, "radians": 0.4},
  {"operation": "loss", "mode": 1, "transmission": 0.9},
  {"operation": "phase_noise", "covariance": [[0.1, 0.02], [0.02, 0.2]], "mean": [0.0, 0.0]},
  {"operation": "unitary", "real": [[0, 1], [1, 0]], "modes": [0, 1]}
]
```

For `unitary`, use `real` and optional equally shaped `imag` arrays. An imaginary zero scalar is the default. Noise stays in its supplied location; it is not moved to the circuit output. Partial distinguishability, custom Kraus instruments and arbitrary code-space construction use the Python API in this version.

Outputs include input configuration, model and versions, source omitted weight, trace history, numerical diagnostics and absolute optical/detector probabilities. The herald output is still unnormalized. All outcomes are listed, including zero-probability entries. PNR saturation is an inclusive overflow bin.

## Memory JSON format

Run `plq memory examples/configs/memory.json`. Fields are `code`, `noise`, `shots`, `rounds`, `seed`, and `final_perfect_round`. Code forms:

```json
{"name": "repetition", "parameters": {"n": 3, "basis": "Z"}}
```

```json
{"file": "examples/configs/custom_code.json"}
```

Paths are relative to the process working directory. A file and a factory cannot be combined. Arbitrary CodeSpace objects belong in the Python dense channel workflow, not the Pauli-frame CLI.

`noise` accepts `px`, `py`, `pz`, `erasure` and `syndrome_flip`. Each is a probability scalar or a vector of the appropriate length (physical qubits for the first four, checks for syndrome flips). Pauli probabilities sum to at most one per site. Pauli events are exclusive, not independent X/Y/Z draws. Noise parameters are per round, not per second. A time-to-error-rate model must be supplied separately.

JSON never evaluates Python or imports arbitrary modules. Unknown fields and physically invalid parameters fail with a nonzero CLI exit status. Custom callable decoders and code factories are registered through Python.

## Error and cost controls

`ResourceLimitError` is a `ValueError` with an explicit dimension/Kraus budget explanation. A dense matrix needs `16*D*D` bytes; operations and eigensolvers require several arrays. Kraus storage limits do not cap all temporary process memory. Reference decoder construction is also bounded by `max_patterns`.

`DecodeFailure` is an explicit decoder outcome counted conservatively in memory experiments. A generic exception is not converted into a successful correction. A zero-probability branch cannot be conditioned. Non-unitary matrices cannot be imported as passive interferometers, and noisy circuits cannot be exported as lossless Perceval circuits.
