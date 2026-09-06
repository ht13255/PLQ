# Parameter reproduction is not experimental reproduction

Every `source-hom` result now contains an `evidence` object. The two supplied
paper configurations declare their sources and assumptions in `provenance`.
The paper benchmark report contains the same machine-readable boundary.

| Evidence category | Meaning | Automatic experiment-reproduction claim |
| --- | --- | --- |
| `simulation` | User-specified model without paper provenance | false |
| `paper_parameter_reproduction` | Computation using cited numerical parameters and declared assumptions; formula checks and sensitivity may accompany it | false |
| `synthetic_validation` | Software comparison against explicitly synthetic counts | false |
| `experimental_data_comparison` | Fixed-model comparison to supplied experimental counts and protocol declarations | false |

The CLI never assigns an `experimental_reproduction` category or turns that
boolean true because a numerical check passes. Matching an equation, reusing a
paper's brightness/g2/overlap, or obtaining a large goodness-of-fit p-value is not
sufficient to establish that an experiment was reproduced.

A real reproduction additionally needs the actual setup/operating point,
calibrations and uncertainty/covariance, source and detector behavior, timing,
raw measurements, identical selection and normalization, and an independently
reviewed correspondence between measured and predicted observables. None of
the existing source-summary scenarios has all of these. The Ding count-budget
comparison remains a scalar rounded-factor check, not a raw-data experiment fit.

## Run an auditable raw-count comparison

```bash
python -m plq compare-experiment examples/experiments/synthetic_manifest.json --output results/comparison.json
```

The included file is visibly **synthetic**, chosen to exercise the pipeline;
no new experimental dataset is bundled. To compare a real dataset, supply:

- A dataset JSON with explicit `origin` (`experimental` or `synthetic`), `shots`,
  `bins` containing occupation/count tuples, and `other_count`.
- A SHA256 of the exact dataset bytes and an optical scenario JSON. Paths are
  relative to the manifest. Scenario and data hashes are retained in the report.
- `observable` equal to `detector_counts` or `photon_occupation`, matched to the
  actual experiment. A photon-occupation simulator output is not automatically
  the same as measured detector counts.
- A protocol identifying an independent trial, `selection=all_trials`, and explicit
  declarations that the model is fixed and the same observable is compared.
- Nonempty calibration descriptions and dataset/protocol source references.

See [the runnable manifest](../examples/experiments/synthetic_manifest.json) and
[its counts schema](../examples/experiments/synthetic_counts.json). `other_count`
includes every outcome not named in `bins`, including failed/vacuum/no-click
trials when applicable. Bin counts + other_count must equal shots; repeated bins,
hash mismatch, unknown fields, incomplete probability, postselected-only inputs,
Monte Carlo predictions and omitted source/thermal weight are rejected. The
model never rescales accepted-only counts to hide collection failures.

The comparison uses multinomial deviance with a streamed parametric bootstrap
(default 2000 replicates, seed 0), appropriate to the declared independent,
mutually exclusive trial categories. The p-value estimates a fixed-model
upper-tail probability; it is not the probability that the model is true.
No fitted parameters, calibration uncertainty integration, correlated/overlapping
coincidence windows, or experimental authenticity verification is provided.
Those require a different likelihood and additional evidence. An observed event
with exactly zero modeled probability is recorded explicitly with p=0 and a
JSON-safe null deviance. Very small nonzero probabilities remain nonzero.
