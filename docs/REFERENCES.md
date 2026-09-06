# Primary references and SDK sources

These sources establish the physical models and interoperability contracts. PLQ is an independent implementation and is not endorsed by the authors or SDK vendors. Documentation was checked during the 2026-09-05/06 implementation session; exact installed versions are in the validation report.

1. [Quandela Perceval documentation](https://perceval.quandela.net/docs/v1.2/index.html) and [unitary components](https://perceval.quandela.net/docs/v1.2/reference/components/unitary_components.html): optical unitary and beam-splitter conventions. PLQ imports the computed numeric unitary and tests it against the SLOS backend.
2. [PennyLane QubitChannel](https://docs.pennylane.ai/en/stable/code/api/pennylane.QubitChannel.html): fixed Kraus-channel interface. PLQ's flagged CPTP map can be inserted into a QNode; operation import is numeric and not differentiable.
3. [Pasqal: programming a neutral-atom QPU](https://docs.pasqal.com/pulser/programming/): clarifies why Pasqal/Pulser is not a photonic circuit backend.
4. [Ivan, Sabapathy and Simon, Operator-sum Representation for Bosonic Gaussian Channels](https://arxiv.org/abs/1012.4266), Phys. Rev. A 84, 042311 (2011): quantum-limited loss/attenuation channel in operator-sum form.
5. [Osca and Vala, Implementation of photon partial distinguishability in a quantum optical circuit simulation](https://arxiv.org/abs/2208.03250): explicit internal wavepacket modes and overlap-based distinguishability.
6. [Tichy, Sampling of partially distinguishable bosons and the relation to the multidimensional permanent](https://arxiv.org/abs/1410.7687), Phys. Rev. A 91, 022316 (2015): multiphoton interference beyond a single scalar overlap. A direct two-permutation formula is used as an independent three-photon test.
7. [Bartolucci et al., Fusion-based quantum computation](https://arxiv.org/abs/2101.09310): physical resource states, fusion outcomes and architecture-specific QEC. PLQ's small Bell primitive does not reproduce a full threshold calculation from this work.
8. [Knill and Laflamme, A Theory of Quantum Error-Correcting Codes](https://arxiv.org/abs/quant-ph/9604034): error-correctability condition tested by `knill_laflamme`.
9. [Ng and Mandayam, A simple approach to approximate quantum error correction based on the transpose channel](https://arxiv.org/abs/0909.0931), Phys. Rev. A 81, 062342 (2010): general transpose recovery; PLQ reports its finite pseudoinverse cutoff and uses an explicit CPTP completion.
10. [Stim source and documentation](https://github.com/quantumlib/Stim): Clifford circuit simulation, detector error models and detector sampling. The supplied circuit defines the physical abstraction.
11. [PyMatching documentation](https://pymatching.readthedocs.io/en/stable/): minimum-weight perfect matching, graphlike detector errors and optional correlated decoding.

The analytic tests, implementation conventions and finite-model limitations are specified in [PHYSICS.md](PHYSICS.md). No hardware performance, optical loss threshold or universal QEC compatibility claim is inferred merely from citing these references.

12. [Iyer and Poulin, Hardness of decoding quantum stabilizer codes](https://arxiv.org/abs/1310.3235): degenerate decoding chooses a stabilizer equivalence class by summing its error probabilities. PLQ implements bounded exhaustive enumeration for the specified independent Pauli model.

The thermal bath and decoding primary sources were checked again for v0.2 on 2026-09-06.
