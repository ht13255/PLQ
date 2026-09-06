# v0.4 validation record

Recorded benchmark time: 2026-09-06T22:43:02.713130+00:00.
Environment: Python 3.12.13, NumPy 2.3.5,
SciPy 1.17.0, pytest 9.1.1, Stim 1.16.0, PyMatching 2.4.0.

## Executed checks

- `OPENBLAS_NUM_THREADS=1 python -m pytest -q`: **128 passed, 8 skipped**.
  Skips are the uninstalled Perceval/PennyLane integrations (six adapter checks
  and two examples). Stim/PyMatching checks ran. The earlier v0.3 full-SDK
  validation remains a separate historical record; it is not relabeled as a
  v0.4 all-SDK run.
- `python scripts/paper_benchmarks.py`: 4 source scenarios, 16 sensitivity runs;
  maximum triad-formula absolute error below 3e-16. Results regenerated with
  evidence categories and full source hashes.
- `python scripts/scaling_benchmarks.py`: small-case density comparison,
  larger sparse calculation, ideal/output-loss/late hardware cases.
- Both new Python examples and the teleportation/comparison CLI examples ran.
  Comparison input is explicitly synthetic. No raw experimental reproduction
  or independent validation of a complete hardware device was performed.

## Measured scaling

| Case | Retained amplitudes | Sparse wall time | Density reference |
| --- | ---: | ---: | --- |
| 6 modes, 3 photons, 3 mesh layers | 48 | 1.182 ms | 8.666 ms; max matrix-element error 3.1e-17 |
| 16 modes, 4 photons, 3 mesh layers | 536 | 11.523 ms | Not allocated; exceeds default dimension budget |

The larger dense total-cutoff dimension is 4845.
One complex128 matrix would require 375,584,400
bytes. That byte count is theoretical. The measured retained sparse dictionary
size is 125,712 bytes under the shallow
object accounting specified in the JSON; neither number is measured peak RSS.
One timing sample in this environment is not a portable speed guarantee.

## Physical regression targets

| Scenario | Accepted detector event | Computational output | Accepted leakage | Late |
| --- | ---: | ---: | ---: | ---: |
| Ideal teleportation | 0.5 | 0.5 | 0 | 0 |
| Output transmission 0.6 | 0.5 | 0.3 | 0.2 | 0 |
| Controller misses buffer deadline | 0 | 0 | 0 | 0.5 |

Tests additionally check all six Pauli-eigenstate inputs and every accepted
pattern, dark-count false heralds with a vacuum resource, state-dependent
acceptance, coherent conditional output, exact channel completeness, and the
absolute (1/2)^3 weight through a three-qubit ideal-recovery example. Sparse
amplitudes are compared with both density propagation and an independent
high-precision permanent. Loss/phase trajectories and interleaved lossy
interference are checked against exact small optical distributions with fixed
seeds and explicit statistical tolerances. Reproduction tests reject altered
data hashes, omitted trials and mismatched protocol declarations, and retain
finite likelihoods even for very small nonzero modeled probabilities.

Machine results: [scaling_hardware.json](../benchmarks/scaling_hardware.json),
[paper_benchmarks.json](../benchmarks/paper_benchmarks.json).
Physical limits: [HARDWARE_BRIDGE.md](HARDWARE_BRIDGE.md).
