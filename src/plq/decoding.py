"""Exhaustive, degeneracy-aware decoding for small independent Pauli models."""
from collections import OrderedDict
from dataclasses import asdict, dataclass
from itertools import product
from math import exp, fsum, log
import numpy as np
from .numerics import integer, ResourceLimitError
from .qec import StabilizerCode, DecodeFailure, parse_pauli, masks_to_pauli


def _probabilities(code, noise):
    from .simulation import MemoryNoise
    noise = noise or MemoryNoise()
    px, py, pz, pe, pm = noise.arrays(code.n, len(code.generators))
    return np.column_stack((1-px-py-pz, px, py, pz)).clip(0, 1), pe, pm


def _patterns(code, probabilities):
    """Enumerate supported words; logarithms retain very unlikely hypotheses."""
    support = [np.flatnonzero(row) for row in probabilities]
    for letters in product(*support):
        x = z = 0
        for q, e in enumerate(letters):
            bit = 1 << (code.n-1-q)
            if e in (1, 2):
                x |= bit
            if e in (2, 3):
                z |= bit
        log_p = fsum(log(probabilities[q, e]) for q, e in enumerate(letters))
        yield x, z, log_p


class MaximumLikelihoodDecoder:
    """Choose the most probable stabilizer coset, summing ALL its Pauli errors.

    Independent, possibly site-dependent I/X/Y/Z probabilities; perfect current
    syndrome. Erasure flags replace the site's distribution by uniform I/X/Y/Z.
    It is optimal for this single-round model, not for arbitrary noisy histories.
    Complexity is exponential and explicitly bounded. No dense code state needed.
    """
    def __init__(self, code, noise=None, *, max_patterns=1000000, max_cache_entries=16):
        if not isinstance(code, StabilizerCode):
            raise TypeError("MaximumLikelihoodDecoder requires a StabilizerCode")
        self.code = code
        self.probabilities, _, pm = _probabilities(code, noise)
        if np.any(pm):
            raise ValueError("MaximumLikelihoodDecoder requires perfect syndrome readout; use a history-aware decoder")
        self.max_patterns = integer(max_patterns, "max_patterns", 1)
        self.max_cache_entries = integer(max_cache_entries, "max_cache_entries", 1)
        self._cache = OrderedDict()
        self._pivots = sorted(code._span.items(), reverse=True)
        self._table(())

    def _coset(self, x, z):
        row = x | (z << self.code.n)
        # Reduce every pivot, even when a non-pivot leading bit is present.
        for pivot, generator in self._pivots:
            if row & (1 << pivot):
                row ^= generator
        return row

    def _table(self, erased):
        if erased in self._cache:
            self._cache.move_to_end(erased)
            return self._cache[erased]
        probabilities = self.probabilities.copy()
        if erased:
            probabilities[list(erased)] = .25
        count = int(np.prod(np.count_nonzero(probabilities, axis=1), dtype=object))
        if count > self.max_patterns:
            raise ResourceLimitError(f"Maximum-likelihood decoding needs {count} Pauli patterns; increase max_patterns explicitly")
        masses, representatives, syndromes = {}, {}, {}
        for x, z, log_p in _patterns(self.code, probabilities):
            key = self._coset(x, z)
            masses[key] = float(np.logaddexp(masses.get(key, -np.inf), log_p))
            if key not in representatives:
                representatives[key] = (x, z)
                syndromes[key] = self.code.syndrome_masks(x, z)
        best = {}
        for key in masses:
            syndrome = syndromes[key]
            if syndrome not in best or masses[key] > masses[best[syndrome]]:
                best[syndrome] = key
        table = {s: masks_to_pauli(*representatives[key], self.code.n) for s, key in best.items()}
        self._cache[erased] = table
        if len(self._cache) > self.max_cache_entries:
            self._cache.popitem(last=False)
        return table

    def decode(self, syndrome, *, erasures=(), history=()):
        syndrome = tuple(syndrome)
        if len(syndrome) != len(self.code.generators) or any(s not in (0, 1) for s in syndrome):
            raise DecodeFailure("Invalid syndrome")
        erased = tuple(sorted(set(integer(q, "erasure location") for q in erasures)))
        if any(q >= self.code.n for q in erased):
            raise ValueError("Erasure location out of range")
        try:
            return self._table(erased)[syndrome]
        except KeyError as exc:
            raise DecodeFailure("Syndrome has zero probability under the decoder's assumed noise") from exc


@dataclass(frozen=True)
class ExactMemoryResult:
    code: str
    decoder: str
    patterns: int
    success_probability: float
    logical_error_rate: float
    decoder_failure_probability: float
    total_probability: float
    noise: dict
    model: str = "complete Pauli enumeration; one round; ideal syndrome and recovery; any non-stabilizer residual fails"

    def to_dict(self):
        return asdict(self)


def exact_pauli_memory(code, noise=None, *, decoder=None, max_patterns=1000000):
    """Exact finite-model one-round rate, with no Monte Carlo sampling error.

    Only independent Pauli noise and ideal syndrome extraction are supported.
    Nonzero erasure/readout noise is rejected; use simulate_memory or Stim.
    The default is degeneracy-aware maximum likelihood. Floating-point error
    and uncertainty in the supplied physical noise parameters remain.
    """
    if not isinstance(code, StabilizerCode):
        raise TypeError("exact_pauli_memory requires a StabilizerCode")
    probabilities, pe, pm = _probabilities(code, noise)
    if np.any(pe) or np.any(pm):
        raise ValueError("Exact Pauli enumeration requires erasure=0 and syndrome_flip=0")
    budget = integer(max_patterns, "max_patterns", 1)
    count = int(np.prod(np.count_nonzero(probabilities, axis=1), dtype=object))
    if count > budget:
        raise ResourceLimitError(f"Exact Pauli enumeration needs {count} patterns; exceeds max_patterns={budget}")
    decoder = decoder or MaximumLikelihoodDecoder(code, noise, max_patterns=budget)
    success, failures, decoder_failures = [], [], []
    corrections = {}
    for x, z, log_p in _patterns(code, probabilities):
        syndrome = code.syndrome_masks(x, z)
        if syndrome not in corrections:
            try:
                _, word, cx, cz = parse_pauli(decoder.decode(syndrome))
                if len(word) != code.n or code.syndrome_masks(cx, cz) != syndrome:
                    raise ValueError("Decoder correction must have n qubits and match the syndrome")
                corrections[syndrome] = (cx, cz)
            except DecodeFailure:
                corrections[syndrome] = None
        correction = corrections[syndrome]
        p = exp(log_p)
        if correction is None:
            decoder_failures.append(p)
            failures.append(p)
        elif code.in_stabilizer(x ^ correction[0], z ^ correction[1]):
            success.append(p)
        else:
            failures.append(p)
    good, bad = fsum(success), fsum(failures)
    from .simulation import MemoryNoise
    serialized = {k: np.asarray(v).tolist() for k, v in asdict(noise or MemoryNoise()).items()}
    return ExactMemoryResult(code.name, type(decoder).__name__, count, good, bad,
                             fsum(decoder_failures), fsum((good, bad)), serialized)
