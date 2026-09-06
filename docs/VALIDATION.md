# Validation report

Recorded: 2026-09-06T00:29:26.946062+00:00

**55 tests passed; 0 failures, 0 errors, 0 skips.**
Editable package installation, all six runnable examples, the CLI, and actual optional SDK integrations were exercised.

## Deterministic comparisons

| Check | Measured result |
| --- | --- |
| Balanced identical-photon HOM coincidence | 0 |
| Each HOM bunching outcome | 0.5000000000000002, 0.5000000000000002 |
| Partial-overlap HOM coincidence, amplitude overlap 0.7 | 0.2550000000000002 (analytic 0.255) |
| Maximum Fock amplitude deviation from a 70-decimal permanent reference | 1.274e-16 |
| Maximum probability deviation from Perceval SLOS | 5.551e-17 |
| Maximum density-entry deviation from PennyLane default.mixed | 1.388e-17 |
| Single-qubit Pauli errors corrected across the 5-, 7-, and 9-qubit codes | 63 / 63 |
| Five-qubit code: Pauli assignments across every pair of erased locations | 160 / 160 |

Additional tests cover vacuum, binomial loss, loss of coherence, unequal dual-rail filtering, Gaussian phase covariance,
three-photon complex Gram phases against an independent permutation formula, saturated detectors, false heralds,
destructive measurement traces, complete Bell instruments, signed/CSS/custom codes, coherent error recovery,
Knill-Laflamme conditions, finite bosonic transpose recovery, resource guards, and JSON/NPZ interchange.

## Sampled examples

Repetition memory: p(X)=0.08, one round, seed 77, 30,000 shots. Analytic block error rate is 0.018176.
Measured 555 failures: 0.01850000, Wilson 95% interval [0.01703568, 0.02008762].

Stim rotated surface-code memory: distance 3, 3 rounds, gate depolarization 0.008 and measurement-flip probability 0.008.
Seed 123, 2,000 shots: 42 any-observable failures, rate 0.02100000.
These rates are examples of their stated models, not a photonic hardware threshold or a comparison of the two architectures.

## Validated versions

| Component | Version |
| --- | --- |
| numpy | 2.5.2 |
| scipy | 1.18.1 |
| pytest | 9.1.1 |
| mpmath | 1.3.0 |
| perceval-quandela | 1.2.4 |
| pennylane | 0.45.1 |
| stim | 1.16.0 |
| pymatching | 2.4.0 |
| python | 3.12.13 |

## Reproduce

```bash
python -m pip install -e '.[all]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/validate.py
```

PowerShell users can set the two environment variables separately before running Python.
The script reruns tests and regenerates this report, `benchmarks/validation.json`, and the directly validated version list.

All numerical deviations are measured on the finite test cases, not global accuracy bounds. Main simulation remains complex128.
The default cutoff/memory guards intentionally prevent some larger exact optical embeddings. Further architecture or hardware claims
require their own physical circuits, calibration data, decoder studies and convergence/error-budget analysis.

The repository includes GitHub Actions configuration. This document records local execution, not an assertion that hosted CI completed.
