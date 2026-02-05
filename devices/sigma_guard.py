"""Device-level Sigma guard wrapper around N and NX.

SigmaGuard is a small utility that applies the Σ-orthogonal projection (N-stage)
to a `MultipolarValue` and, optionally, repeats it multiple times (NX). It is
intended to sit between O2/O3 of a transmitter/receiver cascade where devices
traditionally enforce Σ→0 before decoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from ..core.value import MultipolarValue
from ..physics.sigma import n_stage, nx_stage, sigma_norm


@dataclass
class SigmaGuard:
    """Apply Σ→0 purification (N) and multi-section removal (NX) to values.

    Minimal interface so higher-level code can express the intent clearly:
    - `apply(mv)`: one N-stage.
    - `apply_nx(mv, sections)`: repeated stages; returns all intermediate values.
    - `residual(mv)`: |Σ| after any stage.

    Advanced: set ``linear_coeffs`` to enforce a weighted linear law
    Σ_c = ∑(cᵢ·aᵢ) → 0 instead of plain Σ = ∑ aᵢ.
    """

    sections: int = 1
    linear_coeffs: Sequence[float] | None = None

    def apply(self, mv: MultipolarValue, *, linear_coeffs: Sequence[float] | None = None) -> MultipolarValue:
        """Single N-stage (Σ→0) on `mv`."""

        coeffs = self.linear_coeffs if linear_coeffs is None else linear_coeffs
        return n_stage(mv, linear_coeffs=coeffs)

    def apply_nx(
        self,
        mv: MultipolarValue,
        sections: int | Sequence[float] | None = None,
        *,
        linear_coeffs: Sequence[float] | None = None,
    ) -> List[MultipolarValue]:
        """Run NX with the given section count or taps; default uses `self.sections`.

        When ``sections`` is a sequence, it is treated as per-section tap strengths
        (0 < tap ≤ 1) controlling how strongly the mean component is removed.
        """

        coeffs = self.linear_coeffs if linear_coeffs is None else linear_coeffs
        return nx_stage(mv, sections if sections is not None else self.sections, linear_coeffs=coeffs)

    @staticmethod
    def residual(mv: MultipolarValue, *, linear_coeffs: Sequence[float] | None = None) -> float:
        """Return the absolute Σ residual (|Σ|) for convenience in devices/tests."""

        return sigma_norm(mv, linear_coeffs=linear_coeffs)


__all__ = ["SigmaGuard"]
