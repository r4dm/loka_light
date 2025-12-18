"""Helpers for visualisation / export.

Without depending on plotting libraries this module prepares data structures
for plotting Σ, amplitudes and probabilities in external notebooks or
scripts.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

import numpy as np

from ...physics.multipolar_wave import MultiConjugateFunction

from .circuit import CircuitTelemetry
from .gates import sigma_value


def state_summary(state: MultiConjugateFunction) -> Dict[str, Any]:
    """Compact state view: amplitudes, probabilities and Σ.

    Returns a JSON‑serialisable dictionary.
    """

    amplitudes = state.amplitudes
    power = int(state.n_conjugates)
    probs_raw = np.abs(amplitudes) ** power
    norm = float(probs_raw.sum())
    if norm > 0.0:
        probs = (probs_raw / norm).tolist()
    else:
        probs = [0.0 for _ in range(len(amplitudes))]

    sigma = sigma_value(state)
    return {
        "amplitudes_real": [float(x.real) for x in amplitudes],
        "amplitudes_imag": [float(x.imag) for x in amplitudes],
        "probabilities": probs,
        "sigma": {
            "real": float(sigma.real),
            "imag": float(sigma.imag),
            "abs": float(abs(sigma)),
        },
    }


def telemetry_summary(telemetry: CircuitTelemetry) -> Dict[str, Any]:
    """Compact telemetry representation for a circuit run.

    - Σ by stage is converted to real/imag/abs components;
    - N‑projection residuals and measurements are converted to plain types.
    """

    sigma_items = []
    for z in telemetry.sigma_by_stage:
        sigma_items.append(
            {
                "real": float(z.real),
                "imag": float(z.imag),
                "abs": float(abs(z)),
            }
        )

    n_proj_items = [
        {"stage_index": int(idx), "residual": float(res)}
        for idx, res in telemetry.n_projection_residuals
    ]

    meas_items = []
    for stage_idx, outcome, dist in telemetry.measurements:
        meas_items.append(
            {
                "stage_index": int(stage_idx),
                "outcome": int(outcome),
                "distribution": [float(p) for p in dist],
            }
        )

    return {
        "sigma_by_stage": sigma_items,
        "n_projection_residuals": n_proj_items,
        "measurements": meas_items,
    }


__all__ = [
    "state_summary",
    "telemetry_summary",
]
