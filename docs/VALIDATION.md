# Validation report

Recorded: 2026-09-06T12:12:26.745461+00:00

PLQ 0.2.0; mode: **full**.

**81 passed; 0 failures, 0 errors, 0 skipped; 81 collected.**

The report separates executed checks from unavailable optional SDK checks. Skipped integrations are not counted as passes.

## Deterministic comparisons

| Check | Measured value |
| --- | --- |
| hom coincidence | 0 |
| partial hom coincidence | 0.2550000000000002 |
| partial hom expected | 0.255 |
| max fock amplitude error vs 70 decimal permanent | 1.273811594446465e-16 |
| rare dark click probability | 9.999999999999999e-21 |
| detector 5000 photons probability sum | 1 |
| max probability error vs perceval slos | 5.551115123125783e-17 |
| max density entry error vs pennylane | 1.387778780781446e-17 |

The thermal implementation is also tested against an independent full system-plus-bath creation-operator Fock calculation, including coherences and an untouched spectator mode. Other regression cases cover cutoff growth across layers, rectangular Kraus maps, vacuum/identity limits, rare detector tails, saturated count distributions and explicit resource limits.

The maximum-likelihood decoder is tested against independently enumerated stabilizer-group sums, including multiple logical qubits, site-dependent noise and flagged replacements. All previously supported core examples and code-correction tests are rerun.

## Exact one-round decoder comparison

Five-qubit code; independent per-site p(X)=0.001, p(Y)=0.001, p(Z)=0.15; ideal syndrome measurement and recovery. All 1,024 Pauli patterns are included.

| Decoder | Logical block error probability |
| --- | --- |
| MinimumWeightDecoder | 0.16810259224383 |
| MaximumLikelihoodDecoder | 0.036438461381722 |

These are different recovery choices under the same finite noise model. They are not hardware error rates or thresholds.

## Thermal cutoff convergence

Vacuum input, transmission 0.7, bath mean 0.2. The infinite-bath vacuum probability is 1/1.06.

| Bath cutoff | Omitted weight | Vacuum-probability error | Numerical trace drift |
| --- | --- | --- | --- |
| 4 | 0.000128601 | 2.03905e-05 | 0 |
| 8 | 9.9229e-08 | 3.77759e-09 | 1.11022e-16 |
| 12 | 7.65656e-11 | 6.99552e-13 | 3.33067e-16 |

## Sampled memory

Repetition code, p(X)=0.08, one round: 555 failures / 30000 trials, rate 0.01850000, Wilson 95% interval (np.float64(0.017035679730365197), np.float64(0.020087615310018403)). The analytic rate is 0.018176. Seed 77.

Stim distance-3 rotated memory, three rounds, depolarization and measurement flips 0.008: 42 / 2000 failures, seed 123. This is a separate circuit model.

## Versions

| Package | Version |
| --- | --- |
| plq-sim | 0.2.0 |
| python | 3.12.13 |
| numpy | 2.5.2 |
| scipy | 1.18.1 |
| pytest | 9.1.1 |
| mpmath | 1.3.0 |
| perceval-quandela | 1.2.4 |
| pennylane | 0.45.1 |
| stim | 1.16.0 |
| pymatching | 2.4.0 |

## Reproduce

```bash
python -m pip install -e ".[test,reference]"
python scripts/validate.py --core

# Include actual optional SDK comparisons:
python -m pip install -e ".[all]"
python scripts/validate.py
```

The script sets a single BLAS thread for its pytest subprocess, regenerates this report, `benchmarks/validation.json`, and the direct validated-version list. The full command fails if any tests skip.

Main arithmetic is complex128; the 70-decimal permanent is an independent reference only. Measured deviations are finite test-case results, not global numerical bounds or hardware calibration. Bath omissions, floating-point drift and Monte Carlo intervals quantify different errors.

Hosted runs and their source commits are recorded in [GitHub Actions](https://github.com/ht13255/PLQ/actions). This report describes its generating environment; a local run is not evidence of hosted CI completion.
