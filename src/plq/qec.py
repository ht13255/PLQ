"""Validated code spaces, signed stabilizers, CSS codes, and replaceable decoders."""
from functools import cached_property
from itertools import combinations, product
from typing import Protocol, runtime_checkable
import json
import numpy as np
from .numerics import Precision, ResourceLimitError, density, finite_array, integer
from .channels import Branch, KrausChannel


def parse_pauli(label):
    if not isinstance(label, str) or not label:
        raise ValueError("A Pauli must be a nonempty string")
    sign = -1 if label[0] == "-" else 1
    word = label[1:] if label[0] in "+-" else label
    if not word or any(c not in "IXYZ" for c in word):
        raise ValueError("Only Hermitian signed Pauli words +/-[IXYZ]+ are accepted")
    x = sum((c in "XY") << (len(word)-1-i) for i, c in enumerate(word))
    z = sum((c in "ZY") << (len(word)-1-i) for i, c in enumerate(word))
    return sign, word, x, z


def masks_to_pauli(x, z, n):
    return "".join({(0,0): "I", (1,0): "X", (0,1): "Z", (1,1): "Y"}
                   [((x >> (n-1-i)) & 1, (z >> (n-1-i)) & 1)] for i in range(n))


def pauli_left(label, matrix):
    sign, word, x, z = parse_pauli(label)
    a = np.asarray(matrix)
    if a.shape[0] != 2**len(word):
        raise ValueError("Pauli and matrix dimensions differ")
    i = np.arange(len(a))
    phases = np.array([sign * 1j**word.count("Y") * (-1)**((int(j)&z).bit_count()) for j in i])
    out = np.empty_like(a, dtype=complex)
    out[i ^ x] = phases.reshape((-1,) + (1,)*(a.ndim-1)) * a
    return out


def pauli_matrix(label, *, precision=None):
    p = precision or Precision()
    n = len(parse_pauli(label)[1])
    p.guard(2**n)
    return pauli_left(label, np.eye(2**n, dtype=complex))


def _pivots(rows):
    pivots = {}
    for row in rows:
        while row:
            pivot = row.bit_length()-1
            if pivot in pivots:
                row ^= pivots[pivot]
            else:
                pivots[pivot] = row
                break
    return pivots


class CodeSpace:
    """An arbitrary finite-dimensional isometry. Non-stabilizer/bosonic codes welcome.

    Supplying an isometry does not supply an error model, decoder, or physical encoder.
    """
    def __init__(self, isometry, *, name="custom", precision=None):
        self.precision = precision or Precision()
        self.isometry = finite_array(isometry, 2).copy()
        self.physical_dimension, self.logical_dimension = self.isometry.shape
        self.precision.guard(max(self.isometry.shape))
        if min(self.isometry.shape) < 1 or not np.allclose(self.isometry.conj().T @ self.isometry,
                                        np.eye(self.logical_dimension), atol=self.precision.atol, rtol=0):
            raise ValueError("Code columns must form an orthonormal isometry")
        self.name = str(name)
        self.isometry.flags.writeable = False

    def encode(self, state):
        rho = density(state, self.logical_dimension, subnormalized=True, atol=self.precision.atol)
        return self.isometry @ rho @ self.isometry.conj().T

    def decode(self, state):
        rho = density(state, self.physical_dimension, subnormalized=True, atol=self.precision.atol)
        return Branch(self.isometry.conj().T @ rho @ self.isometry)

    def save(self, path):
        np.savez_compressed(path, isometry=self.isometry, name=np.array(self.name), format_version=np.array(1))

    @classmethod
    def load(cls, path, *, precision=None):
        with np.load(path, allow_pickle=False) as data:
            if data["format_version"].item() != 1:
                raise ValueError("Unsupported code-space format")
            return cls(data["isometry"], name=str(data["name"].item()), precision=precision)


class StabilizerCode:
    def __init__(self, generators, *, logical_x=None, logical_z=None, name="custom stabilizer", precision=None):
        self.generators = tuple(generators)
        if not self.generators:
            raise ValueError("At least one stabilizer is required; use CodeSpace for an unencoded space")
        parsed = [parse_pauli(g) for g in self.generators]
        self.n = len(parsed[0][1])
        if any(len(g[1]) != self.n for g in parsed):
            raise ValueError("All stabilizers must have the same number of qubits")
        self.masks = tuple((x, z) for _, _, x, z in parsed)
        if any(((x&v).bit_count() + (z&u).bit_count()) % 2 for x,z in self.masks for u,v in self.masks):
            raise ValueError("Stabilizers must commute")
        self._span = _pivots([x | (z << self.n) for x,z in self.masks])
        if len(self._span) != len(self.generators):
            raise ValueError("Generators must be independent; remove redundant checks or use the Stim path")
        self.k = self.n - len(self.generators)
        self.name, self.precision = str(name), precision or Precision()
        if (logical_x is None) != (logical_z is None):
            raise ValueError("Supply both logical X and logical Z lists, or neither")
        self.logical_x = None if logical_x is None else tuple(logical_x)
        self.logical_z = None if logical_z is None else tuple(logical_z)
        if self.logical_x is not None:
            if len(self.logical_x) != self.k or len(self.logical_z) != self.k:
                raise ValueError("Exactly k logical X/Z pairs are required")
            ops = [parse_pauli(s) for s in (*self.logical_x, *self.logical_z)]
            if any(len(w) != self.n or any(self.syndrome_masks(x,z)) for _,w,x,z in ops):
                raise ValueError("Logical operators must have n qubits and commute with every stabilizer")
            for i, (_,_,x,z) in enumerate(ops):
                for j, (_,_,u,v) in enumerate(ops):
                    actual = ((x&v).bit_count() + (z&u).bit_count()) % 2
                    expected = int(abs(i-j) == self.k and self.k > 0)
                    if actual != expected:
                        raise ValueError("Logical operators do not have canonical Pauli commutation relations")

    def syndrome_masks(self, x, z):
        return tuple(((x&gz).bit_count() + (z&gx).bit_count()) % 2 for gx,gz in self.masks)

    def syndrome(self, pauli):
        _, word, x, z = parse_pauli(pauli)
        if len(word) != self.n:
            raise ValueError("Error Pauli must have n qubits")
        return self.syndrome_masks(x,z)

    def in_stabilizer(self, x, z):
        """Membership up to global Pauli phase (appropriate for error frames)."""
        row = x | (z << self.n)
        while row:
            pivot = row.bit_length()-1
            if pivot not in self._span:
                return False
            row ^= self._span[pivot]
        return True

    def syndrome_projector(self, syndrome):
        syndrome = tuple(syndrome)
        if len(syndrome) != len(self.generators) or any(s not in (0,1) for s in syndrome):
            raise ValueError("Syndrome must be a binary vector with one bit per generator")
        self.precision.guard(2**self.n)
        p = np.eye(2**self.n, dtype=complex)
        for s, g in zip(syndrome, self.generators):
            p = (p + (-1)**s * pauli_left(g, p)) / 2
        return p

    @cached_property
    def codespace(self):
        p = self.syndrome_projector((0,) * len(self.generators))
        if self.logical_x is None:
            values, vectors = np.linalg.eigh(p)
            v = vectors[:, values > 0.5]
        else:
            p0 = p
            for z in self.logical_z:
                p0 = (p0 + pauli_left(z, p0)) / 2
            norms = np.linalg.norm(p0, axis=0)
            zero = p0[:, int(np.argmax(norms))]
            zero = zero / np.linalg.norm(zero)
            vectors = []
            for bits in product((0,1), repeat=self.k):
                vec = zero.copy()
                for bit, x in zip(bits, self.logical_x):
                    if bit:
                        vec = pauli_left(x, vec)
                vectors.append(vec)
            v = np.column_stack(vectors)
        return CodeSpace(v, name=self.name, precision=self.precision)

    def recovery_channel(self, decoder=None):
        """Ideal projective syndrome extraction and ideal Pauli feed-forward.

        This is a mathematically exact recovery map, not a fault-tolerant optical circuit.
        """
        decoder = decoder or MinimumWeightDecoder(self)
        count = 2**len(self.generators)
        self.precision.guard_kraus(2**self.n,2**self.n,count)
        if count > self.precision.max_kraus:
            raise ResourceLimitError("Recovery syndrome count exceeds max_kraus")
        ops = []
        for syndrome in product((0,1), repeat=len(self.generators)):
            correction = decoder.decode(syndrome)
            if self.syndrome(correction) != syndrome:
                raise ValueError("Decoder correction does not match the observed syndrome")
            ops.append(pauli_left(correction, self.syndrome_projector(syndrome)))
        return KrausChannel(ops, precision=self.precision, name="ideal stabilizer recovery")

    def to_dict(self):
        return {"format": "plq.stabilizer.v1", "name": self.name, "generators": list(self.generators),
                "logical_x": self.logical_x, "logical_z": self.logical_z}

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")

    @classmethod
    def load(cls, path, *, precision=None):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.pop("format", None) != "plq.stabilizer.v1":
            raise ValueError("Unsupported stabilizer schema")
        if set(data) - {"name", "generators", "logical_x", "logical_z"}:
            raise ValueError("Unknown code fields")
        return cls(**data, precision=precision)


def css_code(hx, hz, **kwargs):
    arrays = []
    for h in (hx, hz):
        h = np.asarray(h.toarray() if hasattr(h, "toarray") else h)
        if h.ndim != 2 or h.shape[1] == 0 or not np.all(np.isin(h, (0,1))):
            raise ValueError("CSS check matrices must be binary and two-dimensional")
        arrays.append(h.astype(np.uint8))
    hx, hz = arrays
    if hx.shape[1] != hz.shape[1] or np.any((hx @ hz.T) % 2):
        raise ValueError("CSS requires Hx @ Hz.T = 0 modulo 2")
    generators = ["".join(letter if b else "I" for b in row) for h,letter in ((hx,"X"),(hz,"Z")) for row in h]
    return StabilizerCode(generators, **kwargs)


class DecodeFailure(ValueError):
    pass


@runtime_checkable
class Decoder(Protocol):
    def decode(self, syndrome, *, erasures=(), history=()) -> str: ...


class MinimumWeightDecoder:
    """Reference minimum-weight lookup; generic, bounded, and not maximum likelihood.

    Pauli weight ignores channel bias and degeneracy. Supply a custom decoder for them.
    """
    def __init__(self, code, *, max_weight=None, max_patterns=1000000):
        self.code = code
        max_weight = code.n if max_weight is None else integer(max_weight, "max_weight")
        if max_weight > code.n:
            raise ValueError("max_weight exceeds n")
        max_patterns = integer(max_patterns, "max_patterns", 1)
        self.table = {}
        examined = 0
        for weight in range(max_weight+1):
            for positions in combinations(range(code.n), weight):
                for errors in product("XYZ", repeat=weight):
                    examined += 1
                    if examined > max_patterns:
                        raise ResourceLimitError("Decoder construction exceeds max_patterns")
                    label = ["I"] * code.n
                    for q,e in zip(positions, errors):
                        label[q] = e
                    label = "".join(label)
                    self.table.setdefault(code.syndrome(label), label)
            if len(self.table) == 2**len(code.generators):
                break

    def decode(self, syndrome, *, erasures=(), history=()):
        if erasures:
            raise DecodeFailure("MinimumWeightDecoder does not use erasure flags; select ErasureDecoder")
        try:
            return self.table[tuple(syndrome)]
        except KeyError as exc:
            raise DecodeFailure("Syndrome is outside the decoder lookup table") from exc


class ErasureDecoder:
    """Bounded exhaustive minimum cost: erased sites cost 0, all other sites cost 1.

    A flag denotes a known replacement by a maximally mixed qubit, not |0>.
    Location-dependent decoding is cached. This is not a large-code decoder.
    """
    def __init__(self, code, *, max_patterns=1000000):
        self.code, self.max_patterns = code, integer(max_patterns, "max_patterns", 1)
        self.fallback = MinimumWeightDecoder(code, max_patterns=max_patterns)
        self.cache = {}

    def decode(self, syndrome, *, erasures=(), history=()):
        erased = tuple(sorted(set(integer(q, "erasure location") for q in erasures)))
        if any(q >= self.code.n for q in erased):
            raise ValueError("Erasure location out of range")
        if not erased:
            return self.fallback.decode(syndrome, history=history)
        if erased not in self.cache:
            if 4**self.code.n > self.max_patterns:
                raise ResourceLimitError("Erasure decoding exceeds max_patterns")
            table, costs = {}, {}
            for letters in product("IXYZ", repeat=self.code.n):
                word = "".join(letters)
                cost = sum(c != "I" and q not in erased for q,c in enumerate(letters))
                s = self.code.syndrome(word)
                if s not in costs or cost < costs[s]:
                    table[s], costs[s] = word, cost
            self.cache[erased] = table
        try:
            return self.cache[erased][tuple(syndrome)]
        except KeyError as exc:
            raise DecodeFailure("Invalid syndrome") from exc


def repetition_code(n=3, *, basis="Z", precision=None):
    n = integer(n, "n", 2)
    if basis not in ("X", "Z"):
        raise ValueError("basis must be X or Z")
    generators = ["I"*i + basis*2 + "I"*(n-i-2) for i in range(n-1)]
    lx, lz = ("X"*n, "Z"+"I"*(n-1)) if basis == "Z" else ("X"+"I"*(n-1), "Z"*n)
    return StabilizerCode(generators, logical_x=[lx], logical_z=[lz], name=f"repetition-{basis}-{n}", precision=precision)


def five_qubit_code(*, precision=None):
    return StabilizerCode(["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"], logical_x=["XXXXX"],
                          logical_z=["ZZZZZ"], name="five-qubit [[5,1,3]]", precision=precision)


def steane_code(*, precision=None):
    return StabilizerCode(["IIIXXXX", "IXXIIXX", "XIXIXIX", "IIIZZZZ", "IZZIIZZ", "ZIZIZIZ"],
                          logical_x=["XXXXXXX"], logical_z=["ZZZZZZZ"], name="Steane [[7,1,3]]", precision=precision)


def shor_code(*, precision=None):
    return StabilizerCode(["ZZIIIIIII", "IZZIIIIII", "IIIZZIIII", "IIIIZZIII", "IIIIIIZZI", "IIIIIIIZZ",
                           "XXXXXXIII", "IIIXXXXXX"], logical_x=["ZIIZIIZII"], logical_z=["XXXIIIIII"],
                          name="Shor [[9,1,3]]", precision=precision)


_CODE_FACTORIES = {"repetition": repetition_code, "five_qubit": five_qubit_code, "steane": steane_code, "shor": shor_code}


def register_code(name, factory):
    if not isinstance(name, str) or not name or not callable(factory):
        raise ValueError("A nonempty name and callable factory are required")
    if name in _CODE_FACTORIES:
        raise ValueError("A code with this name is already registered")
    _CODE_FACTORIES[name] = factory


def get_code(name, **kwargs):
    try:
        factory = _CODE_FACTORIES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown code {name!r}; registered: {sorted(_CODE_FACTORIES)}") from exc
    result = factory(**kwargs)
    if not isinstance(result, (CodeSpace, StabilizerCode)):
        raise TypeError("Code factories must return CodeSpace or StabilizerCode")
    return result
