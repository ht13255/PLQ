# Optical scaling: exact amplitudes and explicit sampling

PLQ 0.4 adds opt-in `Circuit.run_sparse` and `Circuit.sample`. Existing `run`
remains the density-matrix reference; no automatic approximation is selected.

| Backend | Input / operations | Output and cost |
| --- | --- | --- |
| `density` / `run` | Pure/mixed Fock states; passive optics, loss, Gaussian phase diffusion, finite thermal baths | Full density matrix. D=binomial(m+N,N); at least 16 D² bytes per complex128 matrix. |
| `sparse` / `run_sparse` | Pure occupations or number superpositions; lossless passive optics | Occupation-to-amplitude dictionary. Local creation-operator recurrences; no lifted D×D matrix. |
| `trajectories` / `sample` | Pure input; passive optics, vacuum loss, Gaussian phase diffusion; optional final detectors | Monte Carlo histogram, shot count, seed, marginal Wilson intervals. One state at a time. |

```python
from plq import Circuit, SparseKet, SparseBudget

circuit = Circuit(4).bs(0, 1).bs(1, 2).phase(2, .2)
ket = circuit.run_sparse([1, 0, 1, 0])
print(ket.probabilities())
shots = circuit.loss(2, .85).sample([1, 0, 1, 0], shots=5000, seed=7)
print(shots.to_dict())
print(shots.interval((0, 0, 0, 0)))  # also supports unobserved outcomes
```

Run [the connected-mesh example](../examples/scalable_optics.py) or:

```bash
python -m plq optics examples/configs/trajectories.json --output results/trajectories.json
OPENBLAS_NUM_THREADS=1 python scripts/scaling_benchmarks.py
```

The benchmark records actual single-process wall times and a small-case density
comparison in [scaling_hardware.json](../benchmarks/scaling_hardware.json). The
larger dense matrix's byte count is a formula, not an allocation or measured RSS.
The dictionary measurement excludes shared integers, mode matrices, interpreter
and peak temporary memory; it is not a whole-process memory comparison.

## Exactness and statistical limits

For a local unitary, normalized creation-operator recurrence expands only its
active modes, preserving spectator occupations. Every nonzero complex128
amplitude is retained, including small amplitudes; there is no magnitude cutoff.
A fixed-N global sector can still contain binomial(m+N-1,N) configurations.
Highly mixing circuits remain combinatorial. A circuit of disconnected small
blocks is much easier than a globally mixing one at the same mode/photon count.
Circuit storage itself still uses dense m×m single-particle matrices per step.

For loss, the environment photon count l is sampled with Born probability
`sum_s |a_s|² binomial(n_s,l) (1-eta)^l eta^(n_s-l)`. Its conditional Kraus state
is propagated into subsequent gates. Vacuum and unsuccessful trials are
included. Gaussian phases are sampled from the declared covariance. Averaging
trajectories converges to these channels; finite-shot histograms are not exact
probabilities, and rare outcomes may require many more shots. No density matrix
is reconstructed just to measure output occupation.

Detectors use the existing per-pulse response, including efficiency, timing
acceptance, Poisson dark counts and saturated PNR bins. They are independent
between pulses. Dead time, afterpulsing and source drift are not inferred.
`interval` gives a marginal 95% Wilson interval; it is not a simultaneous bound
on all bins or a systematic/calibration uncertainty estimate.

`SparseBudget(max_photons=256, max_terms=200000, max_transitions=10000000)` limits
input photons, support/cache entries and recurrence work per trajectory. These
are explicit combinatorial work limits, not a byte-exact RSS cap. Limits raise
`ResourceLimitError`; no term is silently dropped. `SparseKet.to_fock` performs
an explicit conversion subject to `Precision`'s dense budget.

Mixed inputs, thermal baths and conditional surviving-state instruments remain
on the density path. Pure internal-wavepacket-expanded states can be represented
as additional modes explicitly; automatic mixed/distinguishable-source routing
is not implemented. Unsupported sparse JSON options are rejected.
