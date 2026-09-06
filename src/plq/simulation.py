"""Exact small-block logical simulation and phenomenological Pauli-frame memory trials."""
from dataclasses import dataclass, asdict
from statistics import NormalDist
import platform
import numpy as np
from .numerics import density, unitary, probability, integer, embed_operator, finite_array
from .channels import Branch, KrausChannel
from .qec import StabilizerCode, MinimumWeightDecoder, ErasureDecoder, DecodeFailure, parse_pauli


def wilson_interval(failures, shots, confidence=0.95):
    failures, shots = integer(failures, "failures"), integer(shots, "shots", 1)
    if failures > shots or not 0 < confidence < 1:
        raise ValueError("Invalid failure count or confidence")
    z = NormalDist().inv_cdf((1+confidence)/2)
    p = failures/shots
    denom = 1 + z*z/shots
    center = (p+z*z/(2*shots))/denom
    half = z*np.sqrt(p*(1-p)/shots+z*z/(4*shots**2))/denom
    return (max(0., center-half), min(1., center+half))


@dataclass
class MemoryNoise:
    px: float | list = 0.0
    py: float | list = 0.0
    pz: float | list = 0.0
    erasure: float | list = 0.0
    syndrome_flip: float | list = 0.0

    def arrays(self, n, checks):
        result = []
        for name, size in (("px",n),("py",n),("pz",n),("erasure",n),("syndrome_flip",checks)):
            value = finite_array(getattr(self,name))
            if np.any(value.imag != 0):
                raise ValueError(f"{name} must be real")
            value = value.real
            value = np.full(size,float(value)) if value.ndim == 0 else value
            if value.shape != (size,) or not np.all(np.isfinite(value)) or np.any((value<0)|(value>1)):
                raise ValueError(f"{name} must be a probability or a length-{size} probability vector")
            result.append(value)
        if np.any(result[0]+result[1]+result[2] > 1):
            raise ValueError("px + py + pz exceeds one on at least one qubit")
        return result


@dataclass
class MemoryResult:
    shots: int
    failures: int
    decoder_failures: int
    logical_error_rate: float
    wilson_95: tuple
    rounds: int
    seed: int
    code: str
    noise: dict
    model: str
    versions: dict

    def to_dict(self):
        return asdict(self)


def simulate_memory(code, noise=None, *, shots=10000, rounds=1, decoder=None, seed=0, final_perfect_round=True):
    """Pauli-frame trials with ideal stabilizer measurements plus classical readout flips.

    Each erased qubit is ideally replenished as maximally mixed: a uniform I/X/Y/Z
    twirl with its location passed to the decoder. Physical absence, ancilla faults,
    fusion construction and propagation are NOT modeled by this convenience engine.
    Custom decoders receive the entire past (observed syndrome, erased locations)
    history for the current shot. Default decoders only use the current syndrome.
    Any non-stabilizer residual is a block error (state-independent convention).
    """
    if not isinstance(code, StabilizerCode):
        raise TypeError("Pauli-frame trials require StabilizerCode; use LogicalQPU for an arbitrary CodeSpace")
    shots, rounds = integer(shots,"shots",1), integer(rounds,"rounds",1)
    seed = integer(seed,"seed")
    if not isinstance(final_perfect_round, (bool, np.bool_)):
        raise ValueError("final_perfect_round must be Boolean")
    noise = noise or MemoryNoise()
    px,py,pz,pe,pm = noise.arrays(code.n, len(code.generators))
    decoder = decoder or (ErasureDecoder(code) if np.any(pe) else MinimumWeightDecoder(code))
    rng = np.random.default_rng(seed)
    failures, decoder_failures = 0, 0
    # One shot at a time limits memory independently of the number of trials.
    for _ in range(shots):
        x, z, history, aborted = 0, 0, [], False
        for _round in range(rounds):
            draw = rng.random(code.n)
            errors = np.where(draw<px,1,np.where(draw<px+py,2,np.where(draw<px+py+pz,3,0)))
            erased = tuple(int(q) for q in np.flatnonzero(rng.random(code.n)<pe))
            ex, ez = 0, 0
            for q, e in enumerate(errors):
                bit = 1 << (code.n-1-q)
                if e in (1,2): ex ^= bit
                if e in (2,3): ez ^= bit
            x ^= ex
            z ^= ez
            for q in erased:
                e = int(rng.integers(4))
                bit = 1 << (code.n-1-q)
                if e in (1,2): x ^= bit
                if e in (2,3): z ^= bit
            true_s = code.syndrome_masks(x,z)
            observed = tuple(int(s ^ flip) for s,flip in zip(true_s, rng.random(len(pm)) < pm))
            try:
                correction = decoder.decode(observed, erasures=erased, history=tuple(history))
                _, word, cx, cz = parse_pauli(correction)
                if len(word) != code.n:
                    raise ValueError("Decoder returned a correction of the wrong size")
                x ^= cx
                z ^= cz
            except DecodeFailure:
                decoder_failures += 1
                aborted = True
                break
            history.append((observed, erased))
        if not aborted and final_perfect_round:
            true_s = code.syndrome_masks(x,z)
            try:
                correction = decoder.decode(true_s, erasures=(), history=tuple(history))
                _, word, cx, cz = parse_pauli(correction)
                if len(word) != code.n:
                    raise ValueError("Decoder returned a correction of the wrong size")
                x ^= cx
                z ^= cz
            except DecodeFailure:
                decoder_failures += 1
                aborted = True
        failures += int(aborted or not code.in_stabilizer(x,z))
    serialized_noise = {key: np.asarray(value).tolist() for key,value in asdict(noise).items()}
    return MemoryResult(shots, failures, decoder_failures, failures/shots, wilson_interval(failures,shots),
                        rounds, seed, code.name, serialized_noise,
                        "phenomenological Pauli memory; ideal mixed-state erasure replacement; "
                        f"final_perfect_round={final_perfect_round}; decoder failures count as block failures",
                        {"numpy":np.__version__, "python":platform.python_version()})


class LogicalQPU:
    """Small dense encoded register. Fluent operations; no implied hardware compilation."""
    def __init__(self, code):
        self.code = code
        self.space = code.codespace if isinstance(code, StabilizerCode) else code
        zero = np.zeros(self.space.logical_dimension, complex)
        zero[0] = 1
        self.rho = self.space.encode(zero)

    def prepare(self, logical_state):
        self.rho = self.space.encode(logical_state)
        return self

    def prepare_physical(self, physical_state, *, subnormalized=False):
        """Import a physical density matrix, including an explicitly weighted optical branch."""
        self.rho = density(physical_state,self.space.physical_dimension,subnormalized=subnormalized,
                           atol=self.space.precision.atol)
        return self

    def logical_gate(self, matrix):
        """Ideal encoded unitary V U V† + I - VV†; NOT an optical gate synthesis."""
        u = unitary(matrix, self.space.precision.atol)
        if u.shape != (self.space.logical_dimension,)*2:
            raise ValueError("Logical unitary dimension differs from the code")
        v = self.space.isometry
        physical = v @ u @ v.conj().T + np.eye(self.space.physical_dimension) - v @ v.conj().T
        self.rho = physical @ self.rho @ physical.conj().T
        return self

    def channel(self, channel):
        if channel.input_dimension != self.space.physical_dimension or channel.output_dimension != self.space.physical_dimension:
            raise ValueError("Channel must preserve the physical Hilbert-space size; represent leakage explicitly")
        self.rho = channel.apply(self.rho).rho
        return self

    def physical_gate(self, matrix, wires=None):
        if wires is None:
            u = unitary(matrix, self.space.precision.atol)
        else:
            n = self.space.physical_dimension.bit_length()-1
            if 2**n != self.space.physical_dimension:
                raise ValueError("Local qubit gates require a power-of-two physical dimension")
            u = embed_operator(unitary(matrix), wires, n)
        return self.channel(KrausChannel.unitary(u, precision=self.space.precision))

    def local_channel(self, channel, wires):
        n = self.space.physical_dimension.bit_length()-1
        if 2**n != self.space.physical_dimension:
            raise ValueError("Local qubit channels require a power-of-two physical dimension")
        return self.channel(KrausChannel([embed_operator(k, wires, n) for k in channel.operators],
                                        trace_preserving=channel.trace_preserving, precision=self.space.precision))

    def correct(self, *, decoder=None, recovery=None):
        if recovery is None:
            if not isinstance(self.code, StabilizerCode):
                raise ValueError("An arbitrary CodeSpace requires a user-supplied Kraus recovery channel")
            recovery = self.code.recovery_channel(decoder)
        return self.channel(recovery)

    def logical_state(self):
        return self.space.decode(self.rho)

    def fock_state(self, *, precision=None):
        from .optics import DualRail
        n = self.space.physical_dimension.bit_length()-1
        if 2**n != self.space.physical_dimension:
            raise ValueError("Dual-rail export requires a qubit physical space")
        return DualRail(n, precision=precision or self.space.precision).encode(self.rho)


def effective_logical_channel(code, physical_channel, *, recovery=None):
    space = code.codespace if isinstance(code, StabilizerCode) else code
    if recovery is not None:
        physical_channel = physical_channel.then(recovery)
    if (physical_channel.input_dimension, physical_channel.output_dimension) != (space.physical_dimension,)*2:
        raise ValueError("Physical channel and code dimensions differ")
    v = space.isometry
    return KrausChannel([v.conj().T @ k @ v for k in physical_channel.operators], trace_preserving=False,
                        precision=space.precision, name="effective logical success")
