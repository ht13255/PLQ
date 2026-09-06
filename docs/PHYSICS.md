# Physical model and accuracy contract

“Exact” means the specified finite mathematical model, up to floating-point arithmetic. It does not mean exact agreement with a physical device.

## Basis and conventions

For `m` modes and total photon cutoff `N`, PLQ includes every occupation

$$|n_0,\ldots,n_{m-1}\rangle,\quad n_i\ge0,\quad\sum_i n_i\le N.$$

The dimension is $D=\binom{m+N}{N}$. States are ordered by ascending total photon number and recursive weak compositions, exposed as `basis.states` and `basis.index`. Use these indices when importing a matrix. This is not a rectangular per-mode tensor ordering.

The basis is invariant under passive optics and downward closed under loss. A per-mode cutoff of one would incorrectly discard bunching. Initial source configurations outside the total cutoff are omitted with an explicit `omitted_probability`; the retained state stays subnormalized. A boundary population is not itself a truncation error for passive optics. Externally supplied active/CV states require a separate cutoff convergence study.

Qubit ordering is big-endian: qubit 0 is the most significant tensor factor. Dual-rail qubit `q` uses modes `(2*q, 2*q+1)` with $|0\rangle=|1,0\rangle$ and $|1\rangle=|0,1\rangle$.

## Passive optics

The single-particle convention is

$$a_j^\dagger\longrightarrow\sum_i U_{ij}a_i^\dagger.$$

For power transmission $T$ and phase $\phi$, `bs` uses

$$U_{\rm BS}=\begin{pmatrix}\sqrt T&-e^{-i\phi}\sqrt{1-T}\\e^{i\phi}\sqrt{1-T}&\sqrt T\end{pmatrix}.$$

Appending $U_1$ then $U_2$ produces $U_2U_1$. The Fock lift uses normalized creation-operator recurrence with all $\sqrt{n+1}$ factors and input factorials. Its independent permanent expression is

$$\langle\mathbf t|\hat U|\mathbf s\rangle=\frac{\operatorname{perm}(U[\mathbf t,\mathbf s])}{\sqrt{\prod_i t_i!\prod_j s_j!}}.$$

Repeated rows/columns correspond to output/input occupations. Vacuum has amplitude one. `reference.fock_amplitude` evaluates this separately with mpmath and a bounded factorial permanent sum.

For two identical photons at a balanced beam splitter, $P(1,1)=0$ and $P(2,0)=P(0,2)=1/2$. With pure-wavepacket amplitude overlap $\mu$, $P(1,1)=(1-|\mu|^2)/2$. Ideal HOM visibility is $|\mu|^2$, not $\mu$.

## Loss and Gaussian phase noise

Each optical loss step couples a mode to an independent **vacuum environment**. Its Kraus action is

$$L_\ell|n\rangle=\sqrt{\binom n\ell}(1-\eta)^{\ell/2}\eta^{(n-\ell)/2}|n-\ell\rangle.$$

Here $\ell\le n$ and $\eta$ is intensity transmission. Including lower photon-number sectors makes the channel trace preserving. For $n$ photons, survivors follow a binomial distribution. Off-diagonal coherences are evolved by the same channel. See the quantum-limited attenuator operator-sum description in [Ivan, Sabapathy and Simon](https://arxiv.org/abs/1012.4266).

For a dual-rail qubit with unequal rail transmissions, the no-loss logical Kraus operator is $\operatorname{diag}(\sqrt{\eta_0},\sqrt{\eta_1})$. Its success probability and normalized output depend on the input. Only equal rail loss becomes state-independent erasure. PLQ does not infer transmission from wavelength, temperature or distance.

`phase_noise(C, mean)` analytically averages Gaussian phases with mean $\mu$ and real positive-semidefinite covariance $C$:

$$\rho_{\mathbf n,\mathbf m}\mapsto\rho_{\mathbf n,\mathbf m}\exp\left[i(\mathbf n-\mathbf m)^T\mu-\frac12(\mathbf n-\mathbf m)^TC(\mathbf n-\mathbf m)\right].$$

Mean uses radians; covariance uses radians squared. Common-mode fluctuations cancel within a fixed total-number sector. Separate calls are independent layers. Cross-time correlations, non-Gaussian drift and bath spectral structure are not implied by this API.

## Partial distinguishability

`wavepacket_input` describes a product of pure normalized internal photon states with $G_{ij}=\langle\phi_i|\phi_j\rangle$. The full complex Gram matrix must be Hermitian, positive semidefinite and unit-diagonal. A factorization $C^\dagger C=G$ creates photons in explicit orthogonal internal modes. Optical evolution becomes $U\otimes I$; loss has distinct environment modes; spatial phase noise is shared by the internal modes at that port. Internal occupations are summed only for final spatial counting. This follows the explicit internal-mode construction described by [Osca and Vala](https://arxiv.org/abs/2208.03250).

Positive eigenvalues are retained, including values below `Precision.atol`; that tolerance validates input and does not select physical rank. Negative eigenvalues within the input tolerance are treated as numerical residuals, and the reconstruction error is reported as `gram_factorization_residual`. For two unit-diagonal wavepackets the factor uses the stable product `(1-abs(g))*(1+abs(g))`, avoiding loss of relative accuracy in a tiny eigensolver eigenvalue. General many-photon factorizations have only floating-point absolute accuracy; their reported residual is not a rare-event relative-error bound. Repeated input ports require normalization of the symmetrized creation-operator product. This constructor is for pure packets; independent mixed spectra use `mixed_wavepacket_input`. Spatial count probabilities cannot be reinterpreted as a coherent spatial state vector.

`gaussian_gram` assumes equal-width temporal wavefunctions

$$\psi_j(t)\propto\exp[-(t-t_j)^2/(4\sigma^2)]e^{-i\omega_jt}.$$

Thus $\sigma$ is the **intensity** standard deviation in seconds and $\omega$ uses radians/second. With $\Delta t=t_i-t_j$ and $\Delta\omega=\omega_i-\omega_j$,

$$G_{ij}=\exp[-\Delta t^2/(8\sigma^2)-\sigma^2\Delta\omega^2/2+i\Delta\omega(t_i+t_j)/2].$$

Subtract a common carrier to improve conditioning. Frequency-dependent optics and spectrally/time-resolved detection require an additional model.

## Independent mixed internal states

`mixed_wavepacket_input` embeds the tensor product of internal density matrices into the sector with one photon in each distinct input port. All matrices use the same orthonormal spectral/polarization basis. This is an isometry, so trace, coherences and positivity are retained without choosing pure packets that merely fit pairwise data.

For two independent photons, the balanced HOM coincidence is

$$P(1,1)=\frac{1-\operatorname{Tr}(\rho_1\rho_2)}{2}.$$

For a symmetric tritter with three independent photons,

$$P(1,1,1)=\frac{2-\sum_{i<j}\operatorname{Tr}(\rho_i\rho_j)+4\operatorname{Re}\operatorname{Tr}(\rho_1\rho_2\rho_3)}{9}.$$

The pairwise traces cannot generally determine the third-order trace. This formula provides an independent check using [Menssen et al.'s mixed-state treatment](https://arxiv.org/abs/1609.09804). Spectral correlations between photons require a joint input model and are not synthesized by this constructor. Internal degrees of freedom remain unresolved by the spatial detector model.

## Imperfect number sources and HOM observables

Let $\mu=\langle n\rangle$ and $g=\langle n(n-1)\rangle/\mu^2$. Under the explicit assumption $P(n\ge3)=0$,

$$p_2=\frac{g\mu^2}{2},\qquad p_1=\mu-g\mu^2,\qquad p_0=1-\mu+\frac{g\mu^2}{2}.$$

The source helper rejects negative probabilities. These moments alone do not identify a general photon-number distribution, spectral wavefunction or noise mechanism. Extraction efficiency, probability of at least one photon, photon-number mean and a finite-efficiency HBT click ratio have different definitions.

`wavepacket_sources` takes independent number distributions at distinct ports. `same_wavepacket` repeats a source's signal wavepacket for each photon. In `orthogonal_noise`, the two-photon sector has one signal and one extra photon orthogonal to every signal and every other source's noise. The one-photon sector remains a signal. Each branch is the normalized bosonic creation-operator product, then weighted by its absolute source probability. A declared total-photon cutoff removes positive branches with a reported tail and no renormalization.

These hypotheses intentionally require a choice. [Ollivier et al.](https://arxiv.org/abs/2005.01743) shows that extra-photon overlap changes the relation between g2 and HOM visibility. PLQ's conditional P(2) hypothesis is not the paper's weak separable-noise field; their visibility correction is not silently applied here.

An independent operator-moment check is available for arbitrary beam-splitter power transmission $T$, $R=1-T$. With input means $\mu_a,\mu_b$, factorial second moments $f_a,f_b$, and the exchange term $J$,

$$\langle n_0 n_1\rangle=TR(f_a+f_b)+(T^2+R^2)\mu_a\mu_b-2TRJ.$$

For the same-wavepacket model, $J=M\mu_a\mu_b$. For source-specific orthogonal noise with support n<=2, $J=M(p_{1a}+p_{2a})(p_{1b}+p_{2b})$. Independent output loss multiplies this cross moment by $\eta_0\eta_1$. This optical moment is not the threshold click coincidence: the latter applies the full detector response to every spatial occupation, including events containing three or four photons.

`source_hom` reports $1-C_{\parallel}/C_{\perp}$ from two direct two-input runs with identical source statistics and detectors. It also reports absolute coincidences. Intrinsic single-photon overlap, intensity-normalized contrasts and time-bin histogram visibilities must not be interchanged. Paper fixtures state which efficiencies and operating points are combined; missing spectral/pulse calibration is not hidden by fitting.

## Calibrated passive field transfer

A square complex transfer matrix $A$ with singular values at most one defines a vacuum-environment passive attenuation channel. `Circuit.transfer` uses

$$A=U\operatorname{diag}(s)V^\dagger$$

and applies $V^\dagger$, independent losses with photon survival $s_j^2$, then $U$. The loss eigenmodes need not be the physical rails; the resulting map includes coherent mixing and environment information. Applying only $A$ to a ket would omit lost-photon branches and is not equivalent. The implementation is checked against an independently constructed full Halmos unitary dilation with vacuum ancillary modes and an explicit environment trace.

Singular-value excess within the declared absolute tolerance is clipped to one and the actual field reconstruction residual is reported. Larger excess is rejected because amplification needs an active environment model. A complex field matrix, including phase calibration, is required; intensity-only data cannot determine it uniquely. This constructor does not model frequency-dependent scattering, thermal input baths or detector effects on its own.

## Detectors and destructive heralding

Each incident photon is independently registered with effective efficiency $\eta_d$. Optional Gaussian arrival-window acceptance multiplies detector efficiency. `timing_sigma_seconds` describes independent arrival acceptance; it does not automatically model spectral decoherence.

Dark counts are Poisson with mean $\lambda=r_{\rm dark}\Delta t$. Before saturation,

$$P(k|n)=\sum_{j=0}^{\min(n,k)}\binom nj\eta_d^j(1-\eta_d)^{n-j}e^{-\lambda}\frac{\lambda^{k-j}}{(k-j)!}.$$

The final PNR bin includes **every count greater than or equal to `saturation`**, evaluated with the Poisson survival function. Threshold detection gives $P(0|n)=(1-\eta_d)^ne^{-\lambda}$ and $P(1|n)=1-P(0|n)$. No tail is discarded.

`herald` is destructive photon counting, not a square-root nondestructive instrument:

$$\widetilde\rho_R=\sum_{\mathbf n}P(\mathbf k|\mathbf n)\langle\mathbf n|\rho|\mathbf n\rangle_M.$$

Measured number sectors are traced; inter-sector coherences do not survive. The remaining trace is the absolute branch probability, including any incoming retained weight. `conditional_state()` explicitly divides by this weight. Zero-probability conditioning raises. For nested unnormalized branches, do not multiply the incoming probability a second time.

Dead time, afterpulsing, inter-gate correlations and photon-number-dependent efficiency are not built in. Using the same measured inefficiency as both a propagation loss and detector loss double-counts it.

## Bell measurement and resource-state fusion

The unboosted dual-rail analyzer mixes mode pairs `(0,2)` and `(1,3)`. Distinct two-click patterns identify $\Psi^+$ and $\Psi^-$. The $\Phi$ pair is unresolved, so four equally likely Bell inputs give ideal success probability $1/2$.

`bell_instruments` exposes a complete three-outcome destructive CP instrument. Noisy conclusive patterns can be false heralds; success does not imply perfect fidelity. For larger resource states, apply the beam splitters and `herald` to selected modes, then explicitly choose the next circuit from the classical outcome. Boosted ancillas, graph construction, full fusion schedules and decoders must be specified. The architecture-level distinction follows [Bartolucci et al.](https://arxiv.org/abs/2101.09310).

## Codes and ideal recovery

A code is an isometry $V$, with $V^\dagger V=I$. Encoding is $V\rho V^\dagger$; decoding returns the possibly subnormalized matrix $V^\dagger\rho V$. Independent commuting signed stabilizers define

$$P_s=\prod_j\frac{I+(-1)^{s_j}g_j}{2},\qquad\mathcal R(\rho)=\sum_s C_sP_s\rho P_sC_s^\dagger.$$

PLQ rejects dependent or noncommuting generators. Supplied logical X/Z operators must have canonical commutation relations. Without those operators, a numerical orthonormal basis labels the code space; do not assume it matches an external encoder.

Dense recovery assumes ideal projective syndrome extraction and ideal feed-forward. `logical_gate(U)` applies $VUV^\dagger+I-VV^\dagger$, a mathematical encoded unitary without an optical gate decomposition or assigned hardware duration/error.

`knill_laflamme` checks the absolute residual of $V^\dagger E_i^\dagger E_jV=\alpha_{ij}I$ for the specified error set, following [Knill and Laflamme](https://arxiv.org/abs/quant-ph/9604034). Residuals scale with operator normalization and do not cover unspecified errors.

For a physical channel $\mathcal N$, `transpose_recovery` uses $R_i=PE_i^\dagger[\mathcal N(P)]^{-1/2}$ with $P=VV^\dagger$. Its pseudoinverse cutoff is reported, and excluded input support is reset to the first codeword to complete the map to CPTP. This is approximate recovery in general, as studied by [Ng and Mandayam](https://arxiv.org/abs/0909.0931), not a claim of optimal hardware decoding.

## Effective logical channels and flags

For physical Kraus operators $K_a$, computational operators $A_a=V_{out}^\dagger K_aV_{in}$ usually define a trace-decreasing map. Renormalizing its output for every input creates a nonlinear transformation, not a quantum channel.

`flagged()` retains the success block and maps the missing effect $M=I-\sum_aA_a^\dagger A_a$ into one orthogonal failure state. Its Kraus rows use the spectral decomposition of $M$, producing a CPTP completion. Failure is coarse-grained; detailed environment labels are not retained. Input failure/padding states stay invariant.

For a normalized pure target and unnormalized output, report

$$p=\operatorname{Tr}\widetilde\rho,\qquad F_{conditional}=\langle\psi|\widetilde\rho|\psi\rangle/p,\qquad F_{weighted}=\langle\psi|\widetilde\rho|\psi\rangle.$$

Conditional fidelity alone can conceal a tiny success probability.

## Memory trials and statistics

`simulate_memory` samples exclusive I/X/Y/Z errors per qubit and round, then optional flagged replacements. Each erased qubit is ideally replenished as maximally mixed, equivalent to a uniform I/X/Y/Z twirl. Only its location is given to the decoder; its hidden Pauli is not. This differs from retaining a physically absent photon in Fock space.

Observed syndrome bits pass through independent binary symmetric readout channels. Reference lookup decoding minimizes Pauli weight, not a channel-biased or degeneracy-aware likelihood. Erasure decoding assigns zero cost at erased locations and unit cost elsewhere. History is passed to custom decoders; built-ins use the current syndrome only. The optional final perfect round is explicit. Any residual Pauli outside the stabilizer group is a block failure, counting any logical action independent of the chosen input state. Decoder failures are also included conservatively.

The Stim bridge instead follows the user's actual measurement schedule, detector error model and logical observables. PyMatching needs a supported graphlike model; failed decompositions raise. These two engines simulate different experiments unless the user deliberately makes their models agree. Neither automatically translates a photonic resource network into circuit faults.

Independent-shot results include Wilson 95% binomial intervals. Seeds and versions are recorded, but exact RNG sequences across library versions/architectures are not guaranteed. Zero sampled failures is not proof of zero physical failure probability. Statistical intervals do not quantify calibration uncertainty, model mismatch or finite-model truncation.


## Thermal attenuation and controlled truncation

`Circuit.thermal_loss` couples a selected mode to an independent thermal oscillator by the same beam-splitter convention as passive optics. For bath mean occupation $\bar n$, define $q=\bar n/(1+\bar n)$ and retain

$$\tau_K=\sum_{k=0}^K (1-q)q^k|k\rangle\langle k|,\qquad
\epsilon_K=q^{K+1}.$$

PLQ evaluates $\operatorname{Tr}_E[U_\eta(\rho\otimes\tau_K)U_\eta^\dagger]$ without renormalizing $\tau_K$. The system cutoff grows from $N$ to $N+K$, so none of the retained bath's possible output photons are projected away. Later loss or interference acts on those enlarged sectors. The thermal-environment dilation is a bosonic Gaussian attenuator model; see [Ivan, Sabapathy and Simon](https://arxiv.org/abs/1012.4266), especially the noisy-channel discussion.

Implementation: exponentiate the real antisymmetric beam-splitter generator in each conserved total-number sector; then sum sparse Kraus index updates over both incoming and outgoing bath occupations. Regression tests use a separate creation-operator Fock lift on the full system-plus-environment density matrix and trace out the environment. Vacuum-bath, identity and full-replacement limits are checked.

For incoming trace $p$, one interacting step retains trace $p(1-\epsilon_K)$ and omits absolute weight $p\epsilon_K$. The omitted operator is positive. Consequently the trace norm of the missing output is that omitted weight, and it bounds the absolute error of any subsequent probability computed by a fixed trace-nonincreasing instrument, apart from floating-point errors. Multiple independent baths accumulate the reported omitted weights. An explicitly truncated input source contributes its own omitted weight too. These bounds do not apply directly to normalized conditional fidelities or unbounded observables such as photon number.

For a vacuum input, the infinite-bath output has a thermal photon distribution with mean $(1-\eta)\bar n$. The validation example checks this distribution and the finite-bath mean analytically. `tail_tolerance` concerns the bath probability tail, not experimental accuracy. `numerical_trace_error` diagnoses the separate floating-point trace drift. The main engine is still complex128.

The mean occupation must be specified for the actual bath mode; no wavelength, temperature, bandwidth or spectral population is inferred. Thermal bath photons and detector dark counts are distinct mechanisms. Thermal steps on `WavepacketState` are rejected because assigning equal populations to its factorization-dependent internal modes would invent a physical spectral bath model.

## Exact Pauli enumeration and degenerate maximum likelihood

For independent per-site Pauli probabilities, `MaximumLikelihoodDecoder` selects, for each perfect current syndrome, the stabilizer coset with the greatest **sum** of probabilities. Members of a coset have the same action on encoded states. This distinction from choosing a most-probable individual word follows [Iyer and Poulin, Hardness of decoding quantum stabilizer codes](https://arxiv.org/abs/1310.3235).

The implementation reduces binary Pauli masks modulo the full stabilizer span, including all pivots, and accumulates coset masses in log space. It supports multiple logical qubits and nonuniform X/Y/Z rates. Ties use deterministic enumeration order. Current flagged replacements are uniform I/X/Y/Z; cache size and word enumeration are explicitly bounded.

`exact_pauli_memory` enumerates the complete nonzero-support distribution, applies one ideal syndrome/recovery round, and adds probabilities of every residual outside the stabilizer. The calculation has no Monte Carlo error. Tiny logical failure probabilities are accumulated directly instead of subtracting a nearly unit success probability. Float64 underflow/roundoff and imperfect physical calibration still limit accuracy. Explicit stabilizer-group sums, analytic repetition-code rates and flagged replacement cases provide independent tests.

Optimality is restricted to this independent Pauli, ideal-current-syndrome model. The decoder is not a full-history decoder and does not infer coherent optical noise, correlated faults or physical syndrome circuits. `simulate_memory` retains its previous default decoder; the exact-rate API rejects erasure and readout noise rather than silently dropping them. Its illustrative noise parameters are not hardware calibration data.
