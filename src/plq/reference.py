"""Independent, deliberately slow high-precision permanent reference."""
from itertools import permutations
from math import factorial
import numpy as np
from .numerics import integer, ResourceLimitError


def permanent(matrix, *, digits=60, max_photons=9):
    """Factorial reference in mpmath. Raises rather than silently falling back to float64."""
    import mpmath as mp
    digits = integer(digits, "digits", 15)
    # mpmath accepts strings/mpmath numbers for input precision beyond complex128.
    rows = list(matrix)
    n = len(rows)
    if n > max_photons:
        raise ResourceLimitError("Reference permanent exceeds factorial work budget")
    if any(len(row) != n for row in rows):
        raise ValueError("Permanent requires a square matrix")
    with mp.workdps(digits):
        a = [[mp.mpc(v) for v in row] for row in rows]
        return mp.fsum(mp.fprod(a[i][p[i]] for i in range(n)) for p in permutations(range(n)))


def fock_amplitude(unitary, source, target, *, digits=60):
    import mpmath as mp
    s = tuple(integer(x, "source occupation") for x in source)
    t = tuple(integer(x, "target occupation") for x in target)
    if sum(s) != sum(t):
        return mp.mpc(0)
    u = np.asarray(unitary)
    if u.shape != (len(t), len(s)):
        raise ValueError("Mode counts differ")
    rows = [i for i, n in enumerate(t) for _ in range(n)]
    cols = [i for i, n in enumerate(s) for _ in range(n)]
    with mp.workdps(digits):
        sub = [[u[i, j] for j in cols] for i in rows]
        denominator = mp.sqrt(mp.fprod(factorial(x) for x in (*s, *t)))
        return permanent(sub, digits=digits) / denominator
