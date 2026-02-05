"""Sigma projection utilities: P_perp, N-stage, and multi-section (NX).

This module is intentionally minimal and self-contained. It exposes three
concepts that many research flows depend on:

- P_perp(n): orthogonal projector that removes the all-ones component.
- N-stage: single pass that applies P_perp to an amplitude vector (Σ→0).
- NX: chained N-stages (N1…NX). With ideal stages Σ becomes ~0 after the
  first pass; with partial taps (see below) the |Σ| residual decreases
  monotonically across sections.
  For some cascades, N/NX can also enforce a weighted linear form
  Σ_c = ∑(cᵢ·aᵢ) → 0 via ``linear_coeffs``.

All docstrings are kept concise so models can infer the intended usage without
pulling external theory. See also `devices/sigma_guard.py` for an instrument
wrapper that fits the transmitter/receiver cascade.

Note on scope (pseudomultipolar vs volumetric): this module implements the
network (pseudomultipolar) Σ→0 operations used in M/N cascades at nodes O2/O3.
It does not model volumetric (3D) field formation/radiation in a medium. For
volumetric chains, use the oscillator/antenna/receiver devices.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ..core.value import MultipolarValue


def _as_linear_coeffs(linear_coeffs: Sequence[float] | None, n: int) -> np.ndarray:
    if linear_coeffs is None:
        return np.ones(n, dtype=np.float64)
    coeffs = np.asarray(list(linear_coeffs), dtype=np.float64)
    if coeffs.ndim != 1 or coeffs.shape[0] != n:
        raise ValueError("linear_coeffs must be a length-n 1-D sequence")
    if not np.any(coeffs):
        raise ValueError("linear_coeffs must not be all zeros")
    return coeffs


def _as_linear_denom(coeffs: np.ndarray) -> float:
    denom = float(np.sum(coeffs))
    if np.isclose(denom, 0.0, atol=1e-15):
        raise ValueError("linear_coeffs must sum to a non-zero value")
    return denom


def p_perp(n: int) -> np.ndarray:
    """Return the Σ-orthogonal projector P_perp for rank n.

    Definition: P_perp = I - (1/n)·1·1^T. Applying P_perp to a length-n vector
    removes the mean component so that the sum of entries is (numerically) zero.
    """

    if n <= 0:
        raise ValueError("n must be positive")
    eye = np.eye(n, dtype=float)
    ones = np.ones((n, 1), dtype=float)
    return eye - (1.0 / float(n)) * (ones @ ones.T)


def _mv_to_array(mv: MultipolarValue) -> np.ndarray:
    """Pack `MultipolarValue` coefficients into a length-n complex array.

    Order is the loka's polarity order. Missing coefficients are treated as 0.
    """

    return np.asarray(
        [mv.coefficients.get(pol, 0.0) for pol in mv.loka.polarities], dtype=np.complex128
    )


def _array_to_mv(template: MultipolarValue, arr: np.ndarray) -> MultipolarValue:
    """Build a new `MultipolarValue` from an array using the template's loka."""

    coeffs = {pol: complex(arr[i]) for i, pol in enumerate(template.loka.polarities)}
    return MultipolarValue(template.loka, coeffs)


def sigma_residual(mv: MultipolarValue, *, linear_coeffs: Sequence[float] | None = None) -> complex:
    """Return Σ as a complex number (optionally as a weighted linear form).

    - Default: Σ = ∑ aᵢ over all poles (complex).
    - With ``linear_coeffs``: Σ_c = ∑ (cᵢ · aᵢ), useful for linear laws
      Aa+Bb+…→0 in pseudomultipolar cascades.
    """

    vec = _mv_to_array(mv)
    coeffs = _as_linear_coeffs(linear_coeffs, len(vec))
    return complex(np.sum(vec * coeffs))


def sigma_norm(mv: MultipolarValue, *, linear_coeffs: Sequence[float] | None = None) -> float:
    """Return |Σ| (magnitude) for the chosen Σ / linear form.

    This is a simple residual measure: after a correct N-stage, it should be
    close to zero; across NX it should monotonically decrease.
    """

    return float(abs(sigma_residual(mv, linear_coeffs=linear_coeffs)))


def n_stage(mv: MultipolarValue, *, linear_coeffs: Sequence[float] | None = None) -> MultipolarValue:
    """Single N-stage (Σ→0): remove the common component from the coefficient vector.

    The stage preserves the loka and returns a new `MultipolarValue` with the
    mean component removed. Numerically, `sigma_norm(result, linear_coeffs=...)`
    should be ~0.

    Note: when using ``linear_coeffs``, they must sum to a non-zero value.
    """

    vec = _mv_to_array(mv)
    if linear_coeffs is None:
        proj = p_perp(len(vec)) @ vec
        return _array_to_mv(mv, proj)

    coeffs = _as_linear_coeffs(linear_coeffs, len(vec))
    denom = _as_linear_denom(coeffs)
    sigma = complex(np.sum(vec * coeffs))
    phi = sigma / denom
    proj = vec - phi * np.ones(len(vec), dtype=np.complex128)
    return _array_to_mv(mv, proj)


def nx_stage(
    mv: MultipolarValue,
    sections: int | Sequence[float],
    *,
    linear_coeffs: Sequence[float] | None = None,
) -> List[MultipolarValue]:
    """Multi-section N (NX): run `n_stage` repeatedly and return all intermediate results.

    Parameters
    - sections: either the number of sections (int ≥ 1) or a sequence of taps.
      When a sequence is given, each tap (0 < tap ≤ 1) specifies how strongly
      the mean component is removed in that section (tap=1 is the full N-stage).
    - linear_coeffs: optional coefficients for a linear law ∑(cᵢ·aᵢ)→0.
    Expectation: `sigma_norm(outputs[i+1], linear_coeffs=...) <= sigma_norm(outputs[i], linear_coeffs=...)` for taps
    in (0, 1]; with tap=1 the operation is idempotent after the first stage.

    Note: when using ``linear_coeffs``, they must sum to a non-zero value.
    """

    taps: List[float]
    if isinstance(sections, int):
        if sections < 1:
            raise ValueError("sections must be ≥ 1")
        taps = [1.0] * sections
    else:
        taps = [float(x) for x in sections]
        if not taps:
            raise ValueError("sections must be ≥ 1")
        if any(tap <= 0.0 or tap > 1.0 for tap in taps):
            raise ValueError("tap values must satisfy 0 < tap ≤ 1")

    if len(taps) < 1:
        raise ValueError("sections must be ≥ 1")
    out: List[MultipolarValue] = []

    vec = _mv_to_array(mv)
    n = len(vec)
    ones = np.ones(n, dtype=np.complex128)
    coeffs = _as_linear_coeffs(linear_coeffs, n)
    denom = _as_linear_denom(coeffs)
    for tap in taps:
        sigma = complex(np.sum(vec * coeffs))
        phi = sigma / denom
        vec = vec - float(tap) * phi * ones
        out.append(_array_to_mv(mv, vec))
    return out
