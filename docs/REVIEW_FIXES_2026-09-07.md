# Execution-review fixes — 2026-09-07

The supplied PLQ 0.4.0 review evaluated commit
`23f1073302706c2a06dc3525a9f501617a0a2efc`. Its two reproduced defects
were confirmed against that unchanged baseline before applying these fixes.

## Changes

| Review finding | Change | Regression coverage |
| --- | --- | --- |
| Monte Carlo ignored `max_patterns` when `decoder` was omitted | CLI forwards the budget into `simulate_memory`; the Python API passes it to the appropriate default decoder | Omitted and explicit selection; scalar/vector erasure noise; construction and later erasure-table budgets; seeded result equivalence |
| Zero padding inflated the inferred photon cutoff | Validate complete distributions first, then infer cutoff from each source's largest positive-probability index | Padded vacuum and non-vacuum states; interior zeros; tiny positive tails; invalid probabilities; explicit truncation; custom normalization tolerance |

Custom Python decoders remain caller-owned, including objects with false Boolean
values. No default decoder replaces a supplied object. Optical input dictionaries
remain unchanged, and explicit truncation still reports omitted probability.

## Reproduce

From an installed checkout:

```bash
python -m pytest -q tests/test_review_regressions.py
python -m pytest -q
```

The focused suite passed all 34 cases, including actual CLI JSON reads and error
exit status. The original baseline suite passed all 136 tests before changes.
After the fixes, the complete suite passed **170 tests, 0 skipped**, in 8.41
seconds on this Linux environment, including installed optional SDK tests.
This is a single measured test run, not a performance or hardware-accuracy claim.
Numerical/model benchmarks in `benchmarks/` retain their existing provenance;
these bug fixes do not make those older artifacts results of a new experiment.

## Remaining review recommendations

Scenario sweeps with units, plots and assumption summaries; isolated peak-RSS
benchmarks across photon counts and mesh depths; and measured calibration with
hold-out validation remain future work. The latter requires actual experimental
data and a defined hardware/syndrome/decoder interface. These fixes do not add
detector memory, source drift, a fusion-network schedule or a validated mapping
from optical noise to Stim. No hardware-accuracy or threshold claim is made.
