# Paper-parameter benchmarks

Recorded: 2026-09-06T22:43:03.724104+00:00

PLQ 0.4.0. All numbers below were computed by `scripts/paper_benchmarks.py`.

Evidence category: `paper_parameter_reproduction`; `experimental_reproduction=false`. See [the comparison contract](REPRODUCTION.md).

Three papers supply numerical scenarios; a fourth motivates explicit noise hypotheses. Published measurements, derived inputs and scenario assumptions are recorded separately in [paper_parameters.json](../benchmarks/paper_parameters.json). Full machine results, configurations, versions and input hashes are in [paper_benchmarks.json](../benchmarks/paper_benchmarks.json).

## Before and after

The unchanged v0.2 source was executed from commit `c4914874603ee7a26ed2d77973f20a8b10eda1e9`. It already agrees with the pure-state single-photon and triad formulas. The concrete regression was its numerical rank cutoff: positive Gram eigenvalues below `atol` disappeared. The new two-wavepacket factorization also avoids subtractive eigensolver error near unit overlap.

| Overlap amplitude | v0.2 P(1,1) | v0.3 P(1,1) | Analytic reference |
| --- | --- | --- | --- |
| 0.999999999999 | 0 | 9.99977878279e-13 | 9.99977878279e-13 |
| 0.99999999999999 | 0 | 9.99200722163e-15 | 9.99200722163e-15 |

These are floating-point regression cases, not demonstrated experimental precision.

## Source scenarios

[Somaschi et al.](https://arxiv.org/abs/1510.06499) supplies brightness 0.154, g2=0.0028 and corrected overlap 0.9956. This combines abstract summaries; the body/caption does not establish one jointly calibrated operating point for the triplet.

[Ding et al.](https://arxiv.org/abs/1601.00284) supplies overlap 0.985 and g2=0.009. Its scenario mean 0.6336 is derived from preparation*extraction, and 0.216 from three downstream efficiencies. The etalon used for purity/HOM measurements is not included in the rate budget, so this combination is a sensitivity scenario.

Each run keeps vacuum through four-photon sectors, the supplied beam splitter, losses and threshold detection. `same_wavepacket` puts extra photons in the source's signal mode. `orthogonal_noise` uses a source-specific orthogonal extra photon only in P(2). These hypotheses are neither statistical bounds nor uniquely determined by g2.

| Paper | Extra-photon hypothesis | Absolute click coincidence | Click visibility | Moment-formula error |
| --- | --- | --- | --- | --- |
| somaschi_2016 | same_wavepacket | 8.279879649e-05 | 0.9930324768 | 3.25e-19 |
| somaschi_2016 | orthogonal_noise | 8.534358294e-05 | 0.9928183329 | 2.51e-18 |
| ding_2016 | same_wavepacket | 2.426290404e-05 | 0.9764072608 | 1.46e-18 |
| ding_2016 | orthogonal_noise | 2.977838965e-05 | 0.9710441182 | 2.71e-19 |

Click visibility here is 1-C_parallel/C_distinguishable for direct two-input experiments. It is not the papers' corrected overlap or their pulsed-interferometer histogram normalization. Reported uncorrected values are retained as context, not used as equal-observable fit targets.

The JSON also reruns each scenario while varying M and g2 separately by their quoted uncertainties, clipping M to its physical interval. This is local sensitivity, not a confidence interval: joint calibration, covariance and a likelihood are unavailable.

[Ollivier et al.](https://arxiv.org/abs/2005.01743) demonstrates why extra-photon overlap matters. Its weak separable-noise approximation has a different source construction and observable; the benchmark does not equate it with the conditional P(2) models above.

## Measured count-budget comparison

Sequential pure-loss channels plus a threshold detector give **3,658,203.648 counts/s**, against **3,700,000 counts/s** reported by Ding et al. The relative difference is **-1.1296%**. The input factors are independently rounded; no fitting or uncertainty bound is claimed. This check pertains to the pre-etalon count budget.

## Three-photon interference

[Menssen et al.](https://arxiv.org/abs/1609.09804), Eq. (4), predicts P111=(2-3r^2+4r^3 cos(phi))/9 for equal overlap moduli r. The r=0.5 scan holds every pairwise HOM overlap fixed.

| Triad phase / pi | Simulated P111 | Formula P111 | Absolute error |
| --- | --- | --- | --- |
| 0.00 | 0.194444444444 | 0.194444444444 | 2.5e-16 |
| 0.25 | 0.178172598955 | 0.178172598955 | 2.78e-17 |
| 0.50 | 0.138888888889 | 0.138888888889 | 8.33e-17 |
| 0.75 | 0.099605178823 | 0.099605178823 | 9.71e-17 |
| 1.00 | 0.0833333333333 | 0.0833333333333 | 8.33e-17 |
| 1.25 | 0.099605178823 | 0.099605178823 | 5.55e-17 |
| 1.50 | 0.138888888889 | 0.138888888889 | 2.78e-17 |
| 1.75 | 0.178172598955 | 0.178172598955 | 8.33e-17 |
| 2.00 | 0.194444444444 | 0.194444444444 | 1.67e-16 |

For a selected mixed internal state diag(0.7,0.3), the new density-matrix input gives **P111=0.193333333333**, matching the supplement's density-trace formula. Replacing the state by pure packets with the same pairwise HOM overlaps gives **0.225206595618** instead. The selected eigenvalues are a verification example, not a measured spectrum.

## Reproduce

```bash
python -m pip install -e ".[test,reference]"
python -m plq source-hom examples/papers/somaschi_2016.json
python -m plq source-hom examples/papers/ding_2016.json --output results/ding.json
python examples/mixed_wavepackets.py
OPENBLAS_NUM_THREADS=1 python scripts/paper_benchmarks.py
```

Independent tests additionally compare lossy transfer matrices with a full vacuum-environment unitary dilation, and check mixed-state coherences, source cutoffs and photon-number moments. These checks establish implementation agreement within the stated models; they do not validate unmodeled spectral correlations, detector recovery, higher source sectors or entire hardware systems.
