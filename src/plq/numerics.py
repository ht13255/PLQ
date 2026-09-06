"""Validation and explicit resource limits. No automatic state repair."""
from dataclasses import dataclass
from math import isfinite
import numpy as np


class ResourceLimitError(ValueError):
    """The requested exact calculation exceeds an explicit budget."""


@dataclass(frozen=True)
class Precision:
    atol: float = 1e-10
    max_dimension: int = 1024
    max_kraus: int = 4096
    max_kraus_bytes: int = 512 * 2**20

    def __post_init__(self):
        if not isfinite(self.atol) or not 0 < self.atol < 0.01:
            raise ValueError("atol must be finite and between 0 and 0.01")
        integer(self.max_dimension, "max_dimension", minimum=1)
        integer(self.max_kraus, "max_kraus", minimum=1)
        integer(self.max_kraus_bytes, "max_kraus_bytes", minimum=1)

    def guard(self, dimension):
        if dimension > self.max_dimension:
            raise ResourceLimitError(
                f"Hilbert dimension {dimension} exceeds {self.max_dimension}; one complex128 "
                f"matrix needs {16 * dimension**2 / 2**20:.1f} MiB, and operations need several. "
                "Reduce modes/photons or use the stabilizer/Stim path."
            )

    def guard_kraus(self, rows, columns, count):
        self.guard(max(rows, columns))
        if count > self.max_kraus or 16*rows*columns*count > self.max_kraus_bytes:
            raise ResourceLimitError(
                f"Kraus expansion needs {count} matrices / {16*rows*columns*count/2**20:.1f} MiB; "
                "exceeds max_kraus or max_kraus_bytes. Use state propagation or increase the explicit budget."
            )


def integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def probability(value, name="probability"):
    if not np.isscalar(value) or not np.isreal(value) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return float(value)


def finite_array(value, ndim=None):
    a = np.asarray(value, dtype=np.complex128)
    if (ndim is not None and a.ndim != ndim) or not np.all(np.isfinite(a)):
        raise ValueError("Expected a finite array of the requested rank")
    return a


def unitary(value, atol=1e-10):
    a = finite_array(value, 2)
    if a.shape[0] == 0 or a.shape[0] != a.shape[1] or not np.allclose(a.conj().T @ a, np.eye(len(a)), atol=atol, rtol=0):
        raise ValueError("Matrix is not unitary at the requested absolute tolerance")
    return a


def density(value, dimension=None, *, subnormalized=False, atol=1e-10):
    a = finite_array(value)
    if a.ndim == 1:
        a = np.outer(a, a.conj())
    if a.ndim != 2 or a.shape[0] != a.shape[1] or len(a) == 0:
        raise ValueError("Expected a state vector or square density matrix")
    if dimension is not None and a.shape != (dimension, dimension):
        raise ValueError(f"State dimension must be {dimension}")
    if not np.allclose(a, a.conj().T, atol=atol, rtol=0):
        raise ValueError("Density matrix is not Hermitian")
    if np.linalg.eigvalsh(a).min() < -atol:
        raise ValueError("Density matrix is not positive semidefinite")
    trace = np.trace(a)
    if abs(trace.imag) > atol or trace.real < -atol or trace.real > 1 + atol:
        raise ValueError("Density trace must lie in [0, 1]")
    if not subnormalized and abs(trace.real - 1) > atol:
        raise ValueError("State must have trace 1; normalize or explicitly allow subnormalization")
    return a.copy()


def diagnostics(rho):
    return {
        "trace_real": float(np.trace(rho).real),
        "trace_imag": float(np.trace(rho).imag),
        "hermiticity_residual": float(np.linalg.norm(rho - rho.conj().T)),
        "minimum_eigenvalue": float(np.linalg.eigvalsh((rho + rho.conj().T) / 2).min()),
        "purity_unnormalized": float(np.trace(rho @ rho).real),
        "dtype": str(rho.dtype),
    }


def embed_operator(operator, targets, n_qubits):
    """Big-endian tensor order; targets are ordered, not sorted."""
    targets = tuple(targets)
    if len(set(targets)) != len(targets) or any(integer(q, "wire") >= n_qubits for q in targets):
        raise ValueError("Invalid or repeated wire")
    op = finite_array(operator, 2)
    if op.shape != (2**len(targets),) * 2:
        raise ValueError("Operator size does not match target wires")
    d = 2**n_qubits
    out = np.zeros((d, d), complex)
    for col in range(d):
        bits = [(col >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        local_col = sum(bits[q] << (len(targets) - 1 - j) for j, q in enumerate(targets))
        for local_row in range(2**len(targets)):
            row_bits = bits.copy()
            for j, q in enumerate(targets):
                row_bits[q] = (local_row >> (len(targets) - 1 - j)) & 1
            row = sum(b << (n_qubits - 1 - q) for q, b in enumerate(row_bits))
            out[row, col] = op[local_row, local_col]
    return out
