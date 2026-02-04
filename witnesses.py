"""Reference CHSH/CGLMP witnesses (NumPy only).

This module implements a compact "passport of correctness" for the physics-toy
user: numbers that can be independently verified without Torch.

Conventions
-----------
- State: the maximally entangled qudit pair |Φ_d> = (1/√d) Σ_i |i,i>.
- Measurements: two settings per party with d outcomes each, built from Fourier
  bases with standard phase shifts (Collins et al., 2002):
    Alice: α = (0, 1/2)
    Bob:   β = (1/4, -1/4)
- Output: CGLMP value I_d (for d=2 equals CHSH ≈ 2.828).
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

Array = np.ndarray


def _omega(d: int) -> complex:
    return complex(np.exp(2j * np.pi / d))


def _amp_phi_component(
    d: int,
    a: int,
    b: int,
    x: int,
    y: int,
    *,
    alpha: Sequence[float] | None = None,
    beta: Sequence[float] | None = None,
) -> complex:
    """Amplitude ⟨u_x^a| ⟨v_y^b| Φ_d⟩ for the standard CGLMP measurements."""

    if alpha is None:
        alpha = (0.0, 0.5)
    if beta is None:
        beta = (0.25, -0.25)
    alpha_a = float(alpha[int(a)]) % 1.0
    beta_b = float(beta[int(b)]) % 1.0
    w = _omega(d)
    delta = (y + beta_b) - (x + alpha_a)
    exponents = np.arange(d, dtype=float) * delta
    s = np.sum(w ** exponents)
    amp = s / (d * np.sqrt(d))
    return complex(amp)


def cglmp_joint_probs_param(
    d: int,
    *,
    alpha: Sequence[float] | None = None,
    beta: Sequence[float] | None = None,
) -> Dict[Tuple[int, int], Array]:
    """Joint probabilities P(x, y | a, b) for a,b∈{0,1}, x,y∈{0..d-1}."""

    if d < 2:
        raise ValueError("d must be ≥ 2")

    probs: Dict[Tuple[int, int], Array] = {}
    for a in (0, 1):
        for b in (0, 1):
            p = np.zeros((d, d), dtype=float)
            for x in range(d):
                for y in range(d):
                    amp = _amp_phi_component(d, a, b, x, y, alpha=alpha, beta=beta)
                    p[x, y] = float(abs(amp) ** 2)
            s = float(p.sum())
            if s > 0.0:
                p /= s
            probs[(a, b)] = p
    return probs


def cglmp_joint_probs(d: int) -> Dict[Tuple[int, int], Array]:
    """Joint probabilities for the standard phase parameters."""

    return cglmp_joint_probs_param(d)


def _p_equal_shift(p: Array, k: int) -> float:
    """Σ P(x, y) over (x - y) ≡ k (mod d)."""

    d = p.shape[0]
    total = 0.0
    for x in range(d):
        for y in range(d):
            if (x - y) % d == k % d:
                total += float(p[x, y])
    return float(total)


def _p_equal_shift_rev(p: Array, k: int) -> float:
    """Σ P(x, y) over (y - x) ≡ k (mod d)."""

    d = p.shape[0]
    total = 0.0
    for x in range(d):
        for y in range(d):
            if (y - x) % d == k % d:
                total += float(p[x, y])
    return float(total)


def cglmp_value(d: int) -> float:
    """CGLMP witness value I_d (for d=2 equals CHSH)."""

    probs = cglmp_joint_probs(d)
    p00 = probs[(0, 0)]
    p10 = probs[(1, 0)]
    p11 = probs[(1, 1)]
    p01 = probs[(0, 1)]

    total = 0.0
    kmax = (d - 1) // 2
    for k in range(kmax + 1):
        weight = 1.0 - 2.0 * k / (d - 1) if d > 1 else 1.0
        term = (
            _p_equal_shift(p00, k)
            + _p_equal_shift_rev(p10, k + 1)
            + _p_equal_shift(p11, k)
            + _p_equal_shift_rev(p01, k)
            - _p_equal_shift(p00, -k - 1)
            - _p_equal_shift_rev(p10, -k)
            - _p_equal_shift(p11, -k - 1)
            - _p_equal_shift_rev(p01, -k - 1)
        )
        total += weight * term
    return float(total)


def chsh_value_from_probs(probs: Dict[Tuple[int, int], Array]) -> float:
    """CHSH value S for d=2 from the joint probability tables.

    For a fixed probability table there are multiple equivalent CHSH forms
    related by local outcome relabelling. To match the "reference passport"
    expectations we report the maximal absolute CHSH value over the standard
    set of sign variants.
    """

    def corr(p: Array) -> float:
        peq = _p_equal_shift(p, 0)
        return float(peq - (1.0 - peq))

    e00 = corr(probs[(0, 0)])
    e01 = corr(probs[(0, 1)])
    e10 = corr(probs[(1, 0)])
    e11 = corr(probs[(1, 1)])
    candidates = (
        e00 + e01 + e10 - e11,
        e00 + e01 - e10 + e11,
        e00 - e01 + e10 + e11,
        -e00 + e01 + e10 + e11,
    )
    return float(max(abs(val) for val in candidates))


def chsh_value() -> float:
    """Convenience wrapper: CHSH value computed from the standard d=2 tables."""

    return chsh_value_from_probs(cglmp_joint_probs(2))


__all__ = [
    "cglmp_joint_probs",
    "cglmp_joint_probs_param",
    "cglmp_value",
    "chsh_value_from_probs",
    "chsh_value",
]
