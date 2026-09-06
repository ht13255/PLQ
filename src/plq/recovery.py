"""Code-independent Knill-Laflamme diagnostics and transpose/Petz recovery."""
import numpy as np
from .channels import KrausChannel
from .qec import StabilizerCode
from .numerics import finite_array, ResourceLimitError


def _space(code):
    return code.codespace if isinstance(code, StabilizerCode) else code


def knill_laflamme(code, errors):
    """Absolute spectral residual of V† Ei† Ej V = alpha_ij I.

    This tests a specified error SET and scales with its operator normalization.
    It is not a decoder threshold or a state fidelity.
    """
    space = _space(code)
    ev = [finite_array(e,2) @ space.isometry for e in errors]
    if not ev:
        raise ValueError("At least one error is required")
    alpha = np.zeros((len(ev),len(ev)),complex)
    residual = 0.
    for i,a in enumerate(ev):
        for j,b in enumerate(ev):
            overlap = a.conj().T @ b
            alpha[i,j] = np.trace(overlap)/space.logical_dimension
            residual = max(residual,float(np.linalg.norm(overlap-alpha[i,j]*np.eye(space.logical_dimension),ord=2)))
    return {"max_absolute_residual":residual, "satisfies_tolerance":residual <= space.precision.atol,
            "alpha":alpha, "atol":space.precision.atol}


def transpose_recovery(code, noise):
    """CPTP completion of the transpose channel for the maximally mixed code state.

    Ri = P Ei† [N(P)]^(-1/2). Null-space input is reset to the first codeword.
    Eigenvalues <= atol use the pseudoinverse cutoff, which is reported on the
    returned channel. This is an approximate recovery in general, not an optimal
    decoder or a finite-energy GKP error-correction implementation.
    """
    space = _space(code)
    d = space.physical_dimension
    if (noise.input_dimension,noise.output_dimension) != (d,d) or not noise.trace_preserving:
        raise ValueError("Transpose recovery requires a square trace-preserving physical channel")
    p = space.isometry @ space.isometry.conj().T
    image = sum(e @ p @ e.conj().T for e in noise.operators)
    values, vectors = np.linalg.eigh((image+image.conj().T)/2)
    support = values > space.precision.atol
    space.precision.guard_kraus(d,d,len(noise.operators)+int(np.count_nonzero(~support)))
    invroot = (vectors[:,support] / np.sqrt(values[support])) @ vectors[:,support].conj().T
    ops = [p @ e.conj().T @ invroot for e in noise.operators]
    for j in np.flatnonzero(~support):
        ops.append(np.outer(space.isometry[:,0], vectors[:,j].conj()))
    if len(ops) > space.precision.max_kraus:
        raise ResourceLimitError("Transpose recovery exceeds max_kraus")
    result = KrausChannel(ops, precision=space.precision, name="transpose recovery")
    result.pseudoinverse_cutoff = space.precision.atol
    result.discarded_image_eigenvalue_max = float(max([0., *values[~support].tolist()]))
    return result
