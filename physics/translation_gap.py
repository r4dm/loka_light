"""Translation gap demo helpers: n → 2 projection after Σ purification.

The "translation gap" is the simplest explanation for why an n-pole signal can
exist (and be detectable by an n-pole receiver) while a 2-pole sensor reports
~0: the sensor applies an implicit projection that can be misaligned with the
Σ-purified (Σ→0) subspace.

This module provides minimal NumPy utilities to build:
- a "match" projection that captures (almost) all signal energy;
- a "mismatch" projection that is (nearly) orthogonal to the signal;
and to report visibility / loss metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


def sigma_value(vec: Array) -> complex:
    """Σ as the linear sum of amplitudes."""

    v = np.asarray(vec, dtype=np.complex128)
    if v.ndim != 1:
        raise ValueError("vec must be 1-D")
    return complex(v.sum())


def sigma_purify(vec: Array) -> Array:
    """Remove the mean component so that Σ→0 (vector form of the N-stage)."""

    v = np.asarray(vec, dtype=np.complex128)
    if v.ndim != 1:
        raise ValueError("vec must be 1-D")
    n = v.shape[0]
    return (v - v.sum() / float(n)).astype(np.complex128)


def l2_normalize(vec: Array) -> Array:
    v = np.asarray(vec, dtype=np.complex128)
    if v.ndim != 1:
        raise ValueError("vec must be 1-D")
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return (v / norm).astype(np.complex128)


def project(vec: Array, projection: Array) -> Array:
    """Project an n-vector into 2D: y = P x, with P shape (2, n)."""

    x = np.asarray(vec, dtype=np.complex128)
    p = np.asarray(projection, dtype=np.complex128)
    if x.ndim != 1:
        raise ValueError("vec must be 1-D")
    if p.ndim != 2 or p.shape[0] != 2 or p.shape[1] != x.shape[0]:
        raise ValueError("projection must have shape (2, n)")
    return (p @ x).astype(np.complex128)


def visibility(signal: Array, observed: Array) -> float:
    """Energy fraction captured by the projection: ||y||^2 / ||x||^2."""

    x = np.asarray(signal, dtype=np.complex128)
    y = np.asarray(observed, dtype=np.complex128)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("signal and observed must be 1-D")
    e_in = float(np.vdot(x, x).real)
    if e_in == 0.0:
        raise ValueError("signal must have non-zero energy")
    e_out = float(np.vdot(y, y).real)
    return float(e_out / e_in)


def _orthonormal_complement(vec: Array, *, rng: np.random.Generator) -> Array:
    v = l2_normalize(vec)
    n = v.shape[0]
    cand = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex128)
    cand = cand - (v.conj() @ cand) * v
    norm = float(np.linalg.norm(cand))
    if norm == 0.0:
        raise ValueError("failed to construct an orthogonal complement vector")
    return (cand / norm).astype(np.complex128)


def projection_match(signal: Array, *, rng: np.random.Generator | None = None) -> Array:
    """Build a 2×n projection that matches the signal (visibility ≈ 1)."""

    rng = rng or np.random.default_rng()
    v = l2_normalize(signal)
    w = _orthonormal_complement(v, rng=rng)
    p = np.vstack([v.conj(), w.conj()]).astype(np.complex128)
    return p


def projection_mismatch(signal: Array, *, rng: np.random.Generator | None = None) -> Array:
    """Build a 2×n projection orthogonal to the signal (visibility ≈ 0).

    Requires n ≥ 3 (so that the orthogonal complement has dimension ≥ 2).
    """

    rng = rng or np.random.default_rng()
    v = l2_normalize(signal)
    n = v.shape[0]
    if n < 3:
        raise ValueError("mismatch projection requires n ≥ 3")

    w1 = _orthonormal_complement(v, rng=rng)
    cand = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex128)
    cand = cand - (v.conj() @ cand) * v
    cand = cand - (w1.conj() @ cand) * w1
    norm = float(np.linalg.norm(cand))
    if norm == 0.0:
        raise ValueError("failed to construct the second orthogonal vector")
    w2 = (cand / norm).astype(np.complex128)
    p = np.vstack([w1.conj(), w2.conj()]).astype(np.complex128)
    return p


@dataclass(frozen=True)
class TranslationGapResult:
    n: int
    sigma_in: complex
    sigma_after_purify: complex
    visibility_match: float
    visibility_mismatch: float

    @property
    def loss_match(self) -> float:
        return float(1.0 - self.visibility_match)

    @property
    def loss_mismatch(self) -> float:
        return float(1.0 - self.visibility_mismatch)


def translation_gap(
    n: int,
    *,
    seed: int | None = 123,
    sigma_clean: bool = True,
) -> tuple[TranslationGapResult, Array, Array, Array, Array, Array]:
    """Generate a signal and compute match/mismatch visibility for n→2."""

    if n < 3:
        raise ValueError("n must be ≥ 3 for translation_gap")
    rng = np.random.default_rng(seed)
    x = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex128)
    sigma_in = sigma_value(x)
    if sigma_clean:
        x = sigma_purify(x)
    x = l2_normalize(x)
    sigma_after = sigma_value(x)

    p_match = projection_match(x, rng=rng)
    p_mismatch = projection_mismatch(x, rng=rng)
    y_match = project(x, p_match)
    y_mismatch = project(x, p_mismatch)
    vis_match = visibility(x, y_match)
    vis_mismatch = visibility(x, y_mismatch)

    return (
        TranslationGapResult(
            n=int(n),
            sigma_in=sigma_in,
            sigma_after_purify=sigma_after,
            visibility_match=float(vis_match),
            visibility_mismatch=float(vis_mismatch),
        ),
        x,
        p_match,
        p_mismatch,
        y_match,
        y_mismatch,
    )


__all__ = [
    "sigma_value",
    "sigma_purify",
    "l2_normalize",
    "project",
    "visibility",
    "projection_match",
    "projection_mismatch",
    "TranslationGapResult",
    "translation_gap",
]
