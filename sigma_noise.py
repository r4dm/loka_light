"""Σ-consistent vs generic unitary noise (NumPy only).

The noise operators here are intended for the pseudo-multipolar / pseudo-quantum
CPU layer, where Σ is tracked as the *linear* sum of complex amplitudes.

Two constructors are provided:
- `unitary_sigma_consistent`: generates a unitary U that preserves Σ exactly
  (Σ(Uψ) == Σ(ψ)).
- `unitary_generic`: a baseline random unitary exp(i ε H) without Σ constraint.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def unitary_sigma_consistent(
    n: int,
    *,
    rng: np.random.Generator | None = None,
    epsilon: float = 0.2,
) -> Array:
    """Construct a unitary U that preserves Σ (i.e. sum of amplitudes).

    Construction:
    - u0 = 1/√n · [1, 1, ..., 1]; complete to an orthonormal basis Q;
    - apply small phase shifts only in the Σ-orthogonal subspace;
    - U = Q · diag(exp(i phases)) · Q^H.

    The basis completion is built so that every column other than u0 has Σ=0,
    which guarantees `1^T U = 1^T` (hence Σ invariance for any input vector).
    """

    if n < 2:
        raise ValueError("n must be ≥ 2")
    rng = rng or np.random.default_rng()
    eps = float(epsilon)

    u0 = np.ones(n, dtype=np.complex128) / np.sqrt(n)
    a = rng.normal(size=(n, n - 1)) + 1j * rng.normal(size=(n, n - 1))

    # Complex Gram–Schmidt with an explicit Σ=0 constraint (orthogonal to u0).
    for j in range(n - 1):
        v = a[:, j]
        v = v - (u0.conj() @ v) * u0
        for k in range(j):
            v = v - (a[:, k].conj() @ v) * a[:, k]
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            v = rng.normal(size=n) + 1j * rng.normal(size=n)
            v = v - (u0.conj() @ v) * u0
            norm = float(np.linalg.norm(v))
        a[:, j] = v / norm

    q = np.column_stack([u0, a]).astype(np.complex128, copy=False)
    phases = np.zeros(n, dtype=np.float64)
    phases[1:] = eps * rng.normal(size=n - 1)
    diag = np.diag(np.exp(1j * phases))
    u = q @ diag @ q.conj().T
    return u


def unitary_generic(
    n: int,
    *,
    rng: np.random.Generator | None = None,
    epsilon: float = 0.2,
) -> Array:
    """Baseline random unitary noise exp(i ε H) for a random Hermitian H."""

    if n < 2:
        raise ValueError("n must be ≥ 2")
    rng = rng or np.random.default_rng()
    eps = float(epsilon)

    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    h = a + a.conj().T
    vals, vecs = np.linalg.eigh(h)
    phases = np.exp(1j * eps * vals)
    u = vecs @ np.diag(phases) @ vecs.conj().T
    return u


__all__ = [
    "unitary_sigma_consistent",
    "unitary_generic",
]

