"""Lightweight DSL and interpreter for pseudo‑quantum circuits.

A circuit is described as a sequence of stages (gates); each stage applies one
operation to the current :class:`MultiConjugateFunction` state.

Supported gate kinds (minimal set for the CPU backend):
- ``"phase"``      – phase shift (see :func:`apply_phase`);
- ``"unitary"``    – linear / unitary‑like transform (:func:`apply_unitary`);
- ``"n_project"``  – N‑projection with Σ monitoring (:func:`n_project`);
- ``"measure"``    – measurement in the polarity basis
  (:func:`measure_polarity`).

M‑superposition is available as :func:`m_superpose` and can be used before
running the circuit to prepare the input state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Sequence, Tuple

import numpy as np

from ...physics.multipolar_wave import MultiConjugateFunction

from .gates import apply_phase, apply_unitary, n_project, measure_polarity, NProjectionResult, sigma_value


GateKind = Literal["phase", "unitary", "n_project", "measure"]


@dataclass
class GateSpec:
    """Description of a single circuit stage.

    Attributes
    ----------
    kind:
        Gate kind (``"phase"``, ``"unitary"``, ``"n_project"``, ``"measure"``).
    params:
        Parameters for the operation. For built‑in kinds:

        - ``"phase"``: ``{"angle": float | Sequence[float]}``;
        - ``"unitary"``: ``{"matrix": np.ndarray}`` with shape (dim, dim);
        - ``"n_project"``: ``{"projector": np.ndarray, "renormalize": bool}``;
        - ``"measure"``: parameters are not required.
    """

    kind: GateKind
    params: Dict[str, Any]


@dataclass
class Circuit:
    """Container for a list of pseudo‑quantum stages."""

    stages: List[GateSpec]


@dataclass
class CircuitTelemetry:
    """Telemetry collected while running a circuit.

    Attributes
    ----------
    sigma_by_stage:
        Σ values after each stage.
    n_projection_residuals:
        Residuals for ``"n_project"`` stages as ``(stage_index, residual)``.
    measurements:
        List of ``(stage_index, outcome_index, distribution)`` tuples.
    """

    sigma_by_stage: List[complex]
    n_projection_residuals: List[Tuple[int, float]]
    measurements: List[Tuple[int, int, np.ndarray]]


def run_circuit_cpu(initial: MultiConjugateFunction, circuit: Circuit, *, rng: np.random.Generator | None = None) -> Tuple[MultiConjugateFunction, CircuitTelemetry]:
    """Run a circuit over a :class:`MultiConjugateFunction` state on CPU.

    Returns the final state and telemetry: Σ trace, N‑projection residuals and
    measurement outcomes.
    """

    state = initial.copy()
    sigma_trace: List[complex] = []
    n_residuals: List[Tuple[int, float]] = []
    measurements: List[Tuple[int, int, np.ndarray]] = []

    for idx, gate in enumerate(circuit.stages):
        if gate.kind == "phase":
            angle = gate.params.get("angle")
            if angle is None:
                raise ValueError("Parameter 'angle' is required for the 'phase' operation")
            state = apply_phase(state, angle)

        elif gate.kind == "unitary":
            matrix = gate.params.get("matrix")
            if matrix is None:
                raise ValueError("Parameter 'matrix' is required for the 'unitary' operation")
            state = apply_unitary(state, matrix)

        elif gate.kind == "n_project":
            projector = gate.params.get("projector")
            if projector is None:
                raise ValueError("Parameter 'projector' is required for the 'n_project' operation")
            renormalize = bool(gate.params.get("renormalize", True))
            result: NProjectionResult = n_project(state, projector, renormalize=renormalize)
            state = result.state
            n_residuals.append((idx, result.residual))

        elif gate.kind == "measure":
            outcome, dist = measure_polarity(state, rng=rng)
            measurements.append((idx, outcome, dist))

        else:  # pragma: no cover - safeguard against forgotten variants
            raise ValueError(f"Unknown gate type: {gate.kind}")

        sigma_trace.append(sigma_value(state))

    telemetry = CircuitTelemetry(
        sigma_by_stage=sigma_trace,
        n_projection_residuals=n_residuals,
        measurements=measurements,
    )
    return state, telemetry


__all__ = [
    "GateKind",
    "GateSpec",
    "Circuit",
    "CircuitTelemetry",
    "run_circuit_cpu",
]
