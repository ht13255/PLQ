# Public API and configuration

## Core objects

| Object / function | Principal operations |
| --- | --- |
| `Precision` | `atol=1e-10`, `max_dimension=1024`, `max_kraus=4096`, `max_kraus_bytes=512*2**20` |
| `FockBasis(modes, max_photons, precision=...)` | `states`, `index`, `occupations`, `dimension` |
| `FockState` | `ket`, `amplitudes`, `mixture`, `probabilities`, `diagnostics`, `postselect`, `conditional` |
| `Circuit(modes)` | Fluent `bs`, `phase`, `unitary`, `loss`, `thermal_loss`, `phase_noise`; `run(state_or_occupation, precision=...)`, `channel(basis)` |
| `DualRail(n_qubits, basis=..., precision=...)` | `encode`, `decode`, `effective_channel`, `isometry` |
| `Detector` | `response(photons)`, `effective_efficiency`, `outcomes` |
| `herald` | Absolute branch probability, unnormalized remaining state, measured/remaining mode order |
| `wavepacket_input` | Explicit internal-mode state, `through`, `spatial_probabilities`, `detection_probabilities` |
| `KrausChannel` | Validated operators; `apply`, `then`, `unitary`, `flagged` |
| `CodeSpace` | `encode`, `decode`, `save`, `load`; arbitrary finite isometry |
| `StabilizerCode` | `codespace`, `syndrome`, `syndrome_projector`, `recovery_channel`, JSON persistence |
| `LogicalQPU` | `prepare`, `prepare_physical`, `physical_gate`, `logical_gate`, `local_channel`, `channel`, `correct`, `logical_state`, `fock_state` |
| `exact_pauli_memory` | Complete one-round independent-Pauli enumeration; default degenerate maximum likelihood |
| `MaximumLikelihoodDecoder` | Noise-aware stabilizer-coset decoding; known erasure flags; bounded exponential enumeration |
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
| `max_photons` | Optional nonnegative initial total cutoff; inferred from complete finite input support if omitted |
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

Run `plq memory examples/configs/memory.json`. Fields are `code`, `noise`, `method`, `decoder`, `max_patterns`, `shots`, `rounds`, `seed`, and `final_perfect_round`. Code forms:

```json
{"name": "repetition", "parameters": {"n": 3, "basis": "Z"}}
```

```json
{"file": "examples/configs/custom_code.json"}
```

CLI code-file paths are relative to the containing memory JSON file. Direct `memory_scenario(config)` calls retain working-directory semantics unless `base_directory` is supplied. This is a v0.2 change from the old CLI working-directory convention. A file and a factory cannot be combined. Arbitrary CodeSpace objects belong in the Python dense channel workflow, not the Pauli-frame CLI.

`noise` accepts `px`, `py`, `pz`, `erasure` and `syndrome_flip`. Each is a probability scalar or a vector of the appropriate length (physical qubits for the first four, checks for syndrome flips). Pauli probabilities sum to at most one per site. Pauli events are exclusive, not independent X/Y/Z draws. Noise parameters are per round, not per second. A time-to-error-rate model must be supplied separately.

JSON never evaluates Python or imports arbitrary modules. Unknown fields and physically invalid parameters fail with a nonzero CLI exit status. Custom callable decoders and code factories are registered through Python.

## Error and cost controls

`ResourceLimitError` is a `ValueError` with an explicit dimension/Kraus budget explanation. A dense matrix needs `16*D*D` bytes; operations and eigensolvers require several arrays. Kraus storage limits do not cap all temporary process memory. Reference decoder construction is also bounded by `max_patterns`.

`DecodeFailure` is an explicit decoder outcome counted conservatively in memory experiments. A generic exception is not converted into a successful correction. A zero-probability branch cannot be conditioned. Non-unitary matrices cannot be imported as passive interferometers, and noisy circuits cannot be exported as lossless Perceval circuits.

## Thermal attenuation (v0.2)

```python
result = Circuit(1).thermal_loss(
    0, transmission=0.7, mean_photons=0.2,
    tail_tolerance=1e-13,
).run([0])
```

`bath_cutoff` is an optional nonnegative integer. If supplied, it overrides automatic cutoff selection; the actual geometric tail is still reported even if it exceeds `tail_tolerance`. Otherwise the smallest cutoff meeting `tail_tolerance` is chosen (default `1e-12`). `mean_photons=0` uses pure loss. `transmission=1` is identity without a bath truncation or basis enlargement.

At each interacting thermal step, the system's **total** cutoff grows by the bath cutoff. The result state's basis is authoritative. `Circuit.channel` may therefore return rectangular, trace-decreasing Kraus matrices; its `input_basis` and `output_basis` record their meaning. `DualRail.decode` and `DualRail.effective_channel` align their computational embeddings with the enlarged basis. Missing bath weight is not physical erasure; the effective channel's failure completion conservatively includes it along with other missing computational weight.

`OpticalResult.truncation_history` lists each thermal step, bath cutoff, bath tail and absolute omitted weight. `bath_omitted_probability` sums those weights. `numerical_trace_error` compares the actual output trace with the initial trace minus that known omission; it is a diagnostic, not a certified floating-point bound. Optical JSON includes those fields plus `initial_total_cutoff` and `final_total_cutoff`. See `examples/configs/thermal.json`.

Internal-wavepacket evolution rejects thermal steps because the spatial bath parameters do not specify populations of the explicit spectral modes. Use an explicitly populated enlarged mode model for that experiment.

## Exact memory and decoder selection (v0.2)

```json
{
  "method": "exact",
  "code": {"name": "five_qubit"},
  "noise": {"px": 0.001, "py": 0.001, "pz": 0.15},
  "decoder": "maximum_likelihood",
  "max_patterns": 1000000
}
```

`method` defaults to `monte_carlo`, preserving the existing memory-trial behavior. Exact mode accepts only one round, `erasure=0`, and `syndrome_flip=0`; omit `shots`, `seed`, and `final_perfect_round`. In exact mode the default decoder is `maximum_likelihood`; `minimum_weight` and `erasure` can also be explicitly selected for comparisons. In Monte Carlo mode the existing minimum-weight/erasure default remains. The CLI restricts maximum likelihood to exact mode to avoid implying an optimal noisy-history decoder.

Python: `MaximumLikelihoodDecoder(code, noise=None, max_patterns=1000000, max_cache_entries=16)` and `exact_pauli_memory(code, noise=None, decoder=None, max_patterns=1000000)`. These are keyword-only options after `noise`. Decoder noise is independent I/X/Y/Z per site; nonzero syndrome readout flips are rejected. Known erased sites have a uniform replacement distribution. The decoder receives history through the shared protocol but does not use it; optimality applies to the stated current-syndrome model only. A zero-probability syndrome raises `DecodeFailure`.

`ExactMemoryResult` reports enumerated nonzero-support patterns, success probability, logical block error rate, decoder-failure probability, total probability, noise and model. Failure includes every residual outside the stabilizer, irrespective of the encoded input. Failures are summed directly, retaining rare values that would disappear in `1 - success`. No Wilson interval or RNG seed is attached because there is no sampling.

`max_patterns` bounds supported words before enumeration; a full Pauli channel needs `4**n`. Erasure flag tables use bounded LRU caching. `max_dimension` does not constrain this Pauli-frame calculation because it never allocates a dense code state.
