"""Sigma projection utilities: P_perp, N-stage, and multi-section (NX).

This module is intentionally minimal and self-contained. It exposes three
concepts that many research flows depend on:

- P_perp(n): orthogonal projector that removes the all-ones component.
- N-stage: single pass that applies P_perp to an amplitude vector (Σ→0).
- NX: chained N-stages (N1…NX). With ideal stages Σ becomes ~0 after the
  first pass; with partial taps (see below) the |Σ| residual decreases
  monotonically across sections.

All docstrings are kept concise so models can infer the intended usage without
pulling external theory. See also `devices/sigma_guard.py` for an instrument
wrapper that fits the transmitter/receiver cascade.

Note on scope (pseudomultipolar vs volumetric): this module implements the
network (pseudomultipolar) Σ→0 operations used in M/N cascades at nodes O2/O3.
It does not model volumetric (3D) field formation/radiation in a medium. For
volumetric chains, use the oscillator/antenna/receiver devices.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from ..core.value import MultipolarValue


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


def sigma_norm(mv: MultipolarValue) -> float:
    """Return |Σ| for the value in the loka's polarity order (complex magnitude).

    This is a simple residual measure: after a correct N-stage, it should be
    close to zero; across NX it should monotonically decrease.
    """

    vec = _mv_to_array(mv)
    return float(abs(vec.sum()))


def n_stage(mv: MultipolarValue) -> MultipolarValue:
    """Single N-stage (Σ→0): apply P_perp to the coefficient vector.

    The stage preserves the loka and returns a new `MultipolarValue` with the
    mean component removed. Numerically, `sigma_norm(result)` should be ~0.
    """

    vec = _mv_to_array(mv)
    proj = p_perp(len(vec)) @ vec
    return _array_to_mv(mv, proj)


def nx_stage(mv: MultipolarValue, sections: int | Sequence[float]) -> List[MultipolarValue]:
    """Multi-section N (NX): run `n_stage` repeatedly and return all intermediate results.

    Parameters
    - sections: either the number of sections (int ≥ 1) or a sequence of taps.
      When a sequence is given, each tap (0 < tap ≤ 1) specifies how strongly
      the mean component is removed in that section (tap=1 is the full N-stage).
    Expectation: `sigma_norm(outputs[i+1]) <= sigma_norm(outputs[i])` for taps
    in (0, 1]; with tap=1 the operation is idempotent after the first stage.
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
    for tap in taps:
        sigma = vec.sum()
        vec = vec - (tap / float(n)) * sigma * ones
        out.append(_array_to_mv(mv, vec))
    return out
