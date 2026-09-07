# Optical instruments and the remaining QEC hardware gap

PLQ 0.4 implements an event-resolved six-rail teleportation primitive. The input
qubit occupies rails 0,1; a Bell resource occupies 2,3 and 4,5. Beam splitters mix
0 with 2 and 1 with 3; four destructive detectors measure 0..3. Rails 4,5 survive.
The four accepted click patterns retain separate completely positive maps and
receive the appropriate physical X or ZX correction. With an ideal Phi+ resource,
this realizes an identity channel with absolute success probability 1/2.

```bash
python -m plq teleportation examples/configs/teleportation.json
python examples/hardware_teleportation.py
```

The supplied configuration is illustrative. It is not an experiment calibration.
The public Python entry point is `teleportation_instrument(...)`.

| Physical input | Implementation |
| --- | --- |
| Six independent rail transmissions | Photon-loss channels; vacuum sectors retained |
| Measured analyzer field matrix | Optional passive 4×4 `analyzer_transfer`, replacing the two ideal beam splitters; SVD residual reported |
| Imperfect resource | Explicit normalized four-mode `FockState`, total cutoff ≤2, including vacuum and multiphoton configurations; default Phi+ is an ideal assumption |
| Four detectors | Existing efficiency, timing acceptance, dark counts, threshold/PNR model; saturation bin is never discarded |
| Feedforward | Detection + processing + switching latency must fit a declared fixed buffer; late accepted events go to a separate classical failure branch |
| Buffer / switch | Photon survival `switch_transmission * 10^(-loss_db_per_second * buffer_seconds / 10)` on both output rails |
| Buffer phase diffusion | Relative phase variance = variance_per_second × buffer_seconds, plus optional output phase |

The buffer-loss coefficient is a phenomenological dB/s input, not a universal
fiber specification. Convert a measured dB/m loss using the measured group
velocity and actual route; do not substitute intensity loss for field amplitude.
The fixed-latency deadline model is not a shared-device queue, jitter distribution
or a physical clock-rate/throughput guarantee. Latency includes the complete
chosen controller processing path. Physical feedforward is selected here;
architectures using deferred Pauli frames need their own schedule.

## Known classical flags and unobserved leakage

`instrument.apply(rho)` returns all accepted event states, accepted probability,
the computational projection, accepted leakage, rejection and late probability.
Input density matrices are normalized; each output retains absolute probability.

**An accepted Bell signature does not prove that the output photon is present.**
With output transmission 0.6 and all else ideal, accepted probability remains
0.5; computational weight is 0.3, and accepted vacuum weight is 0.2. Dark counts
can also create accepted signatures when a resource is absent. PLQ retains these
states rather than handing the decoder a photon-loss flag it could not observe.

`flagged_channel()` is a CPTP map from a qubit into four event-specific Fock
blocks followed by rejected and late scalar states. Each accepted block retains
vacuum, computational occupation and leakage, including their coherences where
physical. It is a classical/quantum event space, not a qubit tensor register.
The order is `PATTERNS` in `plq.hardware`, then rejected, then late. The receiver's
output is discarded in the last two branches, a declared abort policy.

`computational_channel()` is a TNI 2→2 diagnostic projection that composes with
`LogicalQPU.local_channel`. It is **not** an implemented QND presence check and
must not be presented as an observable success/erasure flag. Applying this
projection at three independent physical qubits gives ideal branch weight
(1/2)^3, not deterministic transmission. The example retains this weight through
ideal small-code recovery. No independent-Pauli approximation is used.

## Numerical and physical validation

Four spanning input states reconstruct each event's unnormalized Choi matrix;
positive eigenvalues are retained. Negative roundoff eigenvalues are removed
only below the hard 1e-10 validity tolerance, and their total removed weight and
Hermiticity residual are reported per event. Larger violations raise. Tests
check six input states and every herald against the ideal teleportation identity,
state-dependent acceptance, attenuation, dark false heralds, timing, CPTP
completeness, and composition into a small logical code without normalization.

## What is still required for hardware-level fault tolerance

- A resource preparation circuit and measured joint spectral/temporal states.
  This primitive assumes a common internal wavepacket; arbitrary mixed resources
  describe occupation noise but do not substitute for distinguishability.
- Detector dead time/afterpulsing, correlated source drift, loss identification,
  switch calibration, controller scheduling and device reuse across pulses.
- An explicit fusion network, check/observable definitions, circuit-level noisy
  syndrome extraction and a decoder consuming only experimentally available data.
- Calibration datasets and held-out experimental comparisons, including systematic
  uncertainty and mismatched measurement/normalization conventions.

The component bridge reduces a specific optical-to-logical modeling gap; it does
not synthesize a complete fault-tolerant machine or reproduce a threshold. The
existing generic Stim adapter remains a user-supplied Clifford/Pauli model;
there is no silent conversion of this coherent instrument into a Stim error rate.

## Primary design references

- [Pittman et al., Demonstration of Feed-Forward Control for Linear Optics Quantum Computation (2002)](https://arxiv.org/abs/quant-ph/0204142): measurement-conditioned optical corrections. The PLQ implementation is its own stated primitive, not a reproduction of that apparatus.
- [Metcalf et al., Quantum teleportation on a photonic chip (2014)](https://arxiv.org/abs/1409.4267): integrated entanglement, Bell analysis and tomography as physical components. No reported fidelity is used as a calibration here.
- [Bartolucci et al., Fusion-based quantum computation, arXiv v1 (2021)](https://arxiv.org/abs/2101.09310v1), Sections II, VII and Appendix B: fusion outcomes, erasure and Pauli faults must be related to a specified network and decoder. Its reported thresholds are not PLQ validation targets. Fixed fusion memories can defer physical feedforward; the present fixed-buffer teleporter is a different schedule.
