# QEC compatibility and extension guide

PLQ accepts explicit mathematical and circuit representations. A new code can be represented when it fits one of these interfaces; its physical encoder, measurement schedule and decoder still need to be specified. No code-name parser can automatically provide those missing semantics.

## Compatibility matrix

| Workflow | Connection | Additional input required |
| --- | --- | --- |
| Independent stabilizer code | `StabilizerCode` | Signed generators; logical basis and appropriate decoder |
| CSS / CSS LDPC | `css_code(Hx, Hz)` | Independent commuting binary checks; scalable decoder for large codes |
| Surface / color code | Small stabilizer model or user `.stim` circuit | Measurements, faults, detectors and observables |
| Fusion / cluster-state photonic code | Fock resource states, Bell/herald instruments, or explicit Stim model | Resource graph, failure behavior, feed-forward and decoder |
| Subsystem / gauge code | Fixed-gauge isometry or scheduled Stim measurements | Noncommuting gauge checks cannot be simultaneous stabilizers |
| Floquet / time-dependent checks | User Stim schedule or explicit measurement branches | Time ordering and detector definitions |
| Non-stabilizer / qudit code | `CodeSpace(V)` and `KrausChannel` | Finite isometry plus physical/recovery channels |
| Binomial / finite bosonic code | Fock-ordered isometry | Noise, recovery and any necessary cutoff study |
| Cat / finite-energy GKP | User finite representation | State construction, convergence, homodyne/non-Gaussian instruments and decoder |
| Perceval | Lossless `ACircuit` | Source, noise, herald and detection settings separately |
| PennyLane | Numeric qubit operations or fixed Kraus map | Explicit wire order; preparation/measurements outside conversion |
| Pasqal / Pulser | No direct bridge in v0.1 | A separate neutral-atom model; exchange arrays only with explicit matching conventions |

## Register a new code

```python
from plq import StabilizerCode, register_code, get_code

def make_my_code():
    return StabilizerCode(["ZZI", "IZZ"], logical_x=["XXX"],
                          logical_z=["ZII"], name="custom repetition")

register_code("custom_repetition", make_my_code)
code = get_code("custom_repetition")
code.save("custom_repetition.json")
restored = StabilizerCode.load("custom_repetition.json")
```

Generators must have equal width. Prefix `-` for a negative check. Only Hermitian `+/-` signs are accepted. Identity and dependent generators are rejected. Redundant measurement checks belong in a scheduled circuit model. Logical X/Z operators are optional; omitting them leaves a numerical code-space basis whose labels should not be assumed to match another encoder.

## Supply a decoder

```python
from plq import MinimumWeightDecoder, MemoryNoise, simulate_memory

class MyDecoder:
    def __init__(self, code):
        self.reference = MinimumWeightDecoder(code)

    def decode(self, syndrome, *, erasures=(), history=()):
        return self.reference.decode(syndrome, erasures=erasures, history=history)

report = simulate_memory(code, MemoryNoise(px=0.01), decoder=MyDecoder(code),
                         shots=1000, seed=7)
```

The wrapper is executable but intentionally delegates its policy. Replace the policy with your own algorithm. `syndrome` contains observed bits; `erasures` contains physical indices, not hidden errors; `history` contains previous `(observed_syndrome, erased_locations)` pairs. Return an `n`-character Pauli word. Raise `DecodeFailure` for a declared failure. Other exceptions propagate. Use an erasure-aware decoder when flags are present.

Dense ideal recovery requires corrections that match the input syndrome. A time-aware memory decoder can intentionally defer correction; the final residual still determines success. An optional final perfect round receives the true syndrome, the past history and an empty erasure set because replacement already occurred in preceding rounds.

## Arbitrary finite codes

```python
import numpy as np
from plq import CodeSpace, KrausChannel, LogicalQPU, transpose_recovery

V = np.array([[1, 0], [0, 1], [0, 0]], dtype=complex)
code = CodeSpace(V, name="qutrit embedding")
noise = KrausChannel([np.eye(3)])
recovery = transpose_recovery(code, noise)
qpu = LogicalQPU(code).prepare([1, 0]).channel(noise).correct(recovery=recovery)
code.save("my_code.npz")
```

`V.conj().T @ V` must equal identity. Rows must follow your physical basis. For bosonic states, use `FockBasis.index`. Truncated non-orthogonal states are rejected; if you choose to orthogonalize them, do so explicitly and document the changed code.

`LogicalQPU.channel` preserves the physical Hilbert-space size. Put leakage states inside that space. Rectangular maps are independently supported by `KrausChannel.apply`; embed them into a larger fixed space when needed. A code-space definition by itself does not provide a fault-tolerant encoder or decoder.

## Instruments and feed-forward

`FockState.postselect(predicate)` is an ideal nondestructive projector. `herald` destructively counts selected modes. To continue an optical branch, pass `branch.remaining_state` into the next circuit, retaining its absolute probability. Alternatively, explicitly normalize and separately track its weight. Never multiply the incoming weight twice when using unnormalized branches.

An arbitrary finite instrument can be a mapping from classical outcomes to trace-nonincreasing `KrausChannel` objects. The union of all outcome Kraus operators must be CPTP. `bell_instruments()` provides a concrete example. Continuous homodyne outcomes and detectors with memory need additional instruments.

## External circuit decoders

```python
import stim
from plq.adapters.stim import simulate_stim

circuit = stim.Circuit.from_file("my_code.stim")
report = simulate_stim(circuit, shots=10000, seed=7)
# Or pass decoder=my_batch_decoder to replace PyMatching.
```

`decode_batch(detections)` returns a binary `(shots, num_observables)` array, including the two-dimensional shape for one observable. Unsupported detector error terms are not discarded. A herald in a Stim circuit must have an appropriate detector/decoder interpretation; it does not automatically acquire the semantics of PLQ's separate ideal-replacement memory engine.

## New backend expectations

Document basis ordering, supported state families and instruments, error assumptions, approximation parameters, and success-probability semantics. Validate the added capability against an analytic result or independent implementation. Agreement on one component does not validate a whole fault-tolerant architecture.

## v0.2 exact reference decoder

`MaximumLikelihoodDecoder(code, noise)` implements the existing decoder protocol and sums stabilizer-coset probabilities. It uses the current perfect syndrome and current flagged replacements; history is accepted but unused. It is not a noisy-history decoder. `exact_pauli_memory` supplies a complete one-round independent-Pauli comparison for custom decoders without Monte Carlo uncertainty. Thermal attenuation returns enlarged Fock bases and may expose rectangular Kraus maps; consumers must use the result/output basis. See [API.md](API.md).
