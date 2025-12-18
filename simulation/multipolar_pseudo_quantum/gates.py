"""CPU primitives for pseudo‑quantum gates over :class:`MultiConjugateFunction`.

The module stays strictly in the pseudo‑multipolar cascade domain:

- states are represented by :class:`loka_light.physics.multipolar_wave.MultiConjugateFunction`;
- gates are linear operations on amplitudes (phases, unitary‑like matrices);
- the M‑stage builds controlled superpositions of several states;
- the N‑stage applies a projection/removal with Σ monitoring (Σ→0);
- measurement interprets ``|ψ|^k`` over poles as a probability distribution,
  where ``k = state.n_conjugates``.

Σ is defined here as the sum of complex amplitudes over all poles; before and
after N‑operations ``sigma_value`` and ``sigma_residual`` expose Σ traces for
tests and diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from ...physics.multipolar_wave import MultiConjugateFunction

Array = np.ndarray


def _as_complex_vector(data: Sequence[complex] | Array) -> Array:
    arr = np.asarray(data, dtype=np.complex128)
    if arr.ndim != 1:
        raise ValueError("Expected a one-dimensional amplitude vector")
    return arr


def sigma_value(state: MultiConjugateFunction | Array) -> complex:
    """Compute Σ for a state as the sum of amplitudes."""

    if isinstance(state, MultiConjugateFunction):
        vec = state.amplitudes
    else:
        vec = _as_complex_vector(state)
    return complex(vec.sum())


def sigma_residual(before: complex, after: complex) -> float:
    """Return the absolute residual ``|Σ_after|`` after an N‑operation."""

    return float(abs(after))


def ensure_sigma_constraint(before: complex, after: complex, tol: float = 1e-8) -> bool:
    """Check that ``|Σ_after| < tol`` for an N‑stage.

    Helper for tests/debugging to assert that the pseudo‑multipolar removal
    stage drives Σ sufficiently close to zero.
    """

    _ = before  # keep before/after signature
    return bool(abs(after) < tol)


def apply_phase(state: MultiConjugateFunction, phase: float | Sequence[float]) -> MultiConjugateFunction:
    """Apply a phase shift to the state amplitudes.

    - If ``phase`` is a scalar, the same shift is applied to all poles.
    - If ``phase`` is a vector, its shape must match the state dimension and
      phases are applied component‑wise.
    """

    amplitudes = state.amplitudes
    if np.isscalar(phase):
        phase_vec = float(phase)
        rotated = amplitudes * np.exp(1j * phase_vec)
    else:
        phase_arr = np.asarray(phase, dtype=np.float64)
        if phase_arr.shape != amplitudes.shape:
            raise ValueError("The phase vector must match the state dimension")
        rotated = amplitudes * np.exp(1j * phase_arr)
    return MultiConjugateFunction(rotated, n_conjugates=state.n_conjugates)


def apply_unitary(state: MultiConjugateFunction, matrix: Array) -> MultiConjugateFunction:
    """Apply a linear (typically unitary‑like) transform to the state.

    ``matrix`` must have shape ``(dim, dim)`` with ``dim == len(state)``. True
    unitarity is not enforced here and is expected to be validated in tests if
    required.
    """

    mat = np.asarray(matrix, dtype=np.complex128)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("'matrix' must be square")
    if mat.shape[0] != state.amplitudes.shape[0]:
        raise ValueError("The 'matrix' shape does not match the state dimension")
    new_vec = mat @ state.amplitudes
    return MultiConjugateFunction(new_vec, n_conjugates=state.n_conjugates)


def m_superpose(states: Sequence[MultiConjugateFunction], weights: Sequence[complex] | None = None, *, normalize: bool = True) -> MultiConjugateFunction:
    """M‑superposition of several states with optional weights.

    All states must share the same dimension and ``n_conjugates``.

    Parameters
    ----------
    states:
        Sequence of input states.
    weights:
        Complex weights for the linear combination. If ``None``, uniform
        weights are used.
    normalize:
        If ``True``, the resulting state is L2‑normalized.
    """

    states = list(states)
    if not states:
        raise ValueError("The state list for M-superposition is empty")

    dim = states[0].amplitudes.shape[0]
    k = states[0].n_conjugates
    for s in states[1:]:
        if s.amplitudes.shape[0] != dim:
            raise ValueError("All states must have the same dimension")
        if s.n_conjugates != k:
            raise ValueError("All states must share the same n_conjugates")

    n_states = len(states)
    if weights is None:
        w = np.ones(n_states, dtype=np.complex128) / n_states
    else:
        w = _as_complex_vector(weights)
        if w.shape[0] != n_states:
            raise ValueError("The number of weights must match the number of states")

    acc = np.zeros(dim, dtype=np.complex128)
    for coeff, s in zip(w, states, strict=False):
        acc += coeff * s.amplitudes

    result = MultiConjugateFunction(acc, n_conjugates=k)
    if normalize:
        result.normalize()
    return result


@dataclass
class NProjectionResult:
    """Result of an N‑projection with Σ telemetry.

    Attributes
    ----------
    state:
        Projected state.
    sigma_before:
        Σ before applying the projector.
    sigma_after:
        Σ after applying the projector.
    residual:
        Numeric residual of Σ after projection.
    """

    state: MultiConjugateFunction
    sigma_before: complex
    sigma_after: complex
    residual: float


def n_project(state: MultiConjugateFunction, projector: Array, *, renormalize: bool = True) -> NProjectionResult:
    """Apply an N‑projection (removal) with Σ monitoring.

    ``projector`` is an arbitrary matrix of shape ``(dim, dim)`` (not
    necessarily an orthogonal projector). Σ is recorded before and after, and
    ``sigma_residual`` is reported in the result.
    """

    dim = state.amplitudes.shape[0]
    proj = np.asarray(projector, dtype=np.complex128)
    if proj.ndim != 2 or proj.shape != (dim, dim):
        raise ValueError("'projector' must have shape (dim, dim)")

    sigma_before = sigma_value(state)
    new_vec = proj @ state.amplitudes
    projected = MultiConjugateFunction(new_vec, n_conjugates=state.n_conjugates)
    if renormalize:
        try:
            projected.normalize()
        except ValueError:
            # zero vector after projection is an acceptable edge case
            pass

    sigma_after = sigma_value(projected)
    residual = sigma_residual(sigma_before, sigma_after)
    return NProjectionResult(
        state=projected,
        sigma_before=sigma_before,
        sigma_after=sigma_after,
        residual=residual,
    )


def measure_polarity(state: MultiConjugateFunction, *, rng: np.random.Generator | None = None) -> Tuple[int, Array]:
    """Simulate a measurement in the polarity basis.

    Returns a pair ``(polarity_index, distribution)``, where
    ``distribution`` is the probability array over all poles. Wavefunction
    collapse (setting one component to 1 and the rest to 0) is left to the
    caller.
    """

    if rng is None:
        rng = np.random.default_rng()

    amplitudes = state.amplitudes
    power = int(state.n_conjugates)
    probs = np.abs(amplitudes) ** power
    norm = float(probs.sum())
    if norm == 0.0:
        raise ValueError("Cannot measure a state with zero norm")
    probs = probs / norm
    index = int(rng.choice(len(probs), p=probs))
    return index, probs


__all__ = [
    "apply_phase",
    "apply_unitary",
    "m_superpose",
    "NProjectionResult",
    "n_project",
    "measure_polarity",
    "sigma_value",
    "sigma_residual",
    "ensure_sigma_constraint",
]
