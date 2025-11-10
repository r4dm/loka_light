"""Sigma projection utilities: P_perp, N-stage, and multi-section (NX).

This module is intentionally minimal and self-contained. It exposes three
concepts that many research flows depend on:

- P_perp(n): orthogonal projector that removes the all-ones component.
- N-stage: single pass that applies P_perp to an amplitude vector (Σ→0).
- NX: repeated N-stage passes; expect monotonic decrease of the Σ-residual.

All docstrings are kept concise so models can infer the intended usage without
pulling external theory. See also `devices/sigma_guard.py` for an instrument
wrapper that fits the transmitter/receiver cascade.
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
      When a sequence is given, its length determines the number of passes; tap
      values are reserved for future weighted variants but are not used here.
    Expectation: `sigma_norm(outputs[i+1]) <= sigma_norm(outputs[i])`.
    """

    if isinstance(sections, int):
        count = sections
    else:
        count = len(list(sections))
    if count < 1:
        raise ValueError("sections must be ≥ 1")
    out: List[MultipolarValue] = []
    cur = mv
    for _ in range(count):
        cur = n_stage(cur)
        out.append(cur)
    return out

