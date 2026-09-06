"""Finite-dimensional completely positive maps, including heralded maps."""
from dataclasses import dataclass
import numpy as np
from .numerics import Precision, finite_array, density, probability, unitary

PAULI = {"I": np.eye(2, dtype=complex), "X": np.array([[0, 1], [1, 0]], complex),
         "Y": np.array([[0, -1j], [1j, 0]], complex), "Z": np.diag([1, -1]).astype(complex)}


@dataclass
class Branch:
    """The matrix retains its absolute weight; conditioning is always explicit."""
    rho: np.ndarray

    @property
    def probability(self):
        return float(np.trace(self.rho).real)

    def conditional(self):
        if self.probability <= 0:
            raise ValueError("Cannot condition on a zero-probability branch")
        return self.rho / self.probability


class KrausChannel:
    def __init__(self, operators, *, trace_preserving=True, precision=None, name="custom"):
        self.precision = precision or Precision()
        self.operators = tuple(finite_array(k, 2).copy() for k in operators)
        if not self.operators:
            raise ValueError("At least one Kraus operator is required")
        if len(self.operators) > self.precision.max_kraus:
            from .numerics import ResourceLimitError
            raise ResourceLimitError("Kraus count exceeds max_kraus")
        shape = self.operators[0].shape
        if min(shape) < 1 or any(k.shape != shape for k in self.operators):
            raise ValueError("Kraus operators must share a nonempty shape")
        self.output_dimension, self.input_dimension = shape
        self.precision.guard_kraus(*shape, len(self.operators))
        self.precision.guard(max(shape))
        effect = sum(k.conj().T @ k for k in self.operators)
        residual = np.eye(shape[1]) - effect
        self.completeness_residual = float(np.linalg.norm(residual))
        if trace_preserving:
            if not np.allclose(effect, np.eye(shape[1]), atol=self.precision.atol, rtol=0):
                raise ValueError("Kraus map is not trace preserving")
        elif np.linalg.eigvalsh(residual).min() < -self.precision.atol:
            raise ValueError("Kraus map increases trace")
        self.trace_preserving, self.name = trace_preserving, name
        for k in self.operators:
            k.flags.writeable = False

    def apply(self, state):
        rho = density(state, self.input_dimension, subnormalized=True, atol=self.precision.atol)
        return Branch(sum(k @ rho @ k.conj().T for k in self.operators))

    def then(self, other):
        if self.output_dimension != other.input_dimension:
            raise ValueError("Channel dimensions do not compose")
        if len(self.operators) * len(other.operators) > self.precision.max_kraus:
            from .numerics import ResourceLimitError
            raise ResourceLimitError("Channel composition exceeds max_kraus")
        self.precision.guard_kraus(other.output_dimension, self.input_dimension,
                                  len(self.operators)*len(other.operators))
        return KrausChannel([b @ a for b in other.operators for a in self.operators],
                            trace_preserving=self.trace_preserving and other.trace_preserving,
                            precision=self.precision, name=f"{self.name} -> {other.name}")

    @classmethod
    def unitary(cls, matrix, *, precision=None):
        p = precision or Precision()
        return cls([unitary(matrix, p.atol)], precision=p, name="unitary")

    def flagged(self):
        """Embed a square trace-decreasing map in a qubit space with a failure flag.

        Input lives in indices [0,d). Failure is index d; padding is invariant.
        If d is a power of two, the first (most significant) wire is the flag.
        """
        d = self.input_dimension
        if self.output_dimension != d:
            raise ValueError("Flagging currently requires a square map")
        size = 1 << d.bit_length()
        self.precision.guard(size)
        self.precision.guard_kraus(size, size, len(self.operators)+d+1)
        ops = []
        for k in self.operators:
            a = np.zeros((size, size), complex)
            a[:d, :d] = k
            ops.append(a)
        missing = np.eye(d) - sum(k.conj().T @ k for k in self.operators)
        eigenvalues, vectors = np.linalg.eigh((missing + missing.conj().T) / 2)
        for j, value in enumerate(eigenvalues):
            if value > 0:
                a = np.zeros((size, size), complex)
                a[d, :d] = np.sqrt(value) * vectors[:, j].conj()
                ops.append(a)
        padding = np.zeros((size, size), complex)
        padding[d:, d:] = np.eye(size - d)
        ops.append(padding)
        channel = KrausChannel(ops, precision=self.precision, name=f"flagged({self.name})")
        channel.success_dimension, channel.failure_index = d, d
        return channel


def pauli_channel(px=0.0, py=0.0, pz=0.0):
    weights = [probability(x) for x in (px, py, pz)]
    if sum(weights) > 1:
        raise ValueError("px + py + pz must not exceed 1")
    return KrausChannel([np.sqrt(p) * PAULI[a] for a, p in zip("IXYZ", [1-sum(weights), *weights])], name="Pauli")


def erasure_channel(p):
    """Qubit -> qutrit: |e> is orthogonal to both computational basis states."""
    p = probability(p)
    k0 = np.zeros((3, 2), complex)
    k0[:2] = np.sqrt(1-p) * np.eye(2)
    k1, k2 = np.zeros((3, 2), complex), np.zeros((3, 2), complex)
    k1[2, 0] = k2[2, 1] = np.sqrt(p)
    return KrausChannel([k0, k1, k2], name="erasure")
