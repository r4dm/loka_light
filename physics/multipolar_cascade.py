"""Simplified phasor cascades for multipolar formation/removal stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..core.algebras import LokaCn
from ..core.value import MultipolarValue


@dataclass
class CascadeNodes:
    """Complex potentials measured at O₁/O₂/O₃ in the cascade."""

    phi_o1: complex
    phi_o2: complex
    phi_o3: complex


class PhasorCascade:
    """Utility helpers to assemble multipolar cascade values."""

    @staticmethod
    def build_loka(n: int) -> LokaCn:
        names = [f"P{k}" for k in range(n)]
        return LokaCn(n=n, operation_type="add", loka_name=f"Cn_add_{n}", polarity_names=names)

    @staticmethod
    def build_mv(
        stage: Literal["pseudo", "final"],
        *,
        n: int,
        amplitude: float | complex = 1.0,
        mutual_strength: float = 0.08,
    ) -> MultipolarValue:
        cascade = PseudoMultipolarCascade(n, amplitude=amplitude, mutual_strength=mutual_strength)
        if stage == "pseudo":
            return cascade.build_mv()
        pure = PureMultipolarCascade(n, amplitude=amplitude, mutual_strength=mutual_strength)
        return pure.build_mv()

    @staticmethod
    def is_zero(value: MultipolarValue, *, tol: float = 1e-9) -> bool:
        return abs(value.collapse()) < tol


class BaseCascade:
    """Base class implementing shared phasor utilities."""

    def __init__(
        self,
        n: int,
        *,
        amplitude: float | complex = 1.0,
        mutual_strength: float = 0.08,
    ) -> None:
        if n < 2:
            raise ValueError("n must be ≥ 2")
        self.n = n
        self.amplitude = amplitude
        self.mutual_strength = mutual_strength
        self.loka = PhasorCascade.build_loka(n)

    @property
    def _phasor(self) -> np.ndarray:
        idx = np.arange(self.n, dtype=np.float64)
        phases = np.exp(1j * 2.0 * np.pi * idx / self.n)
        return np.asarray(phases, dtype=np.complex128)

    def nodes(self) -> CascadeNodes:
        phasors = self._phasor * self.amplitude
        sigma = np.sum(phasors)
        compensated = phasors - sigma / self.n
        return CascadeNodes(
            phi_o1=phasors[0],
            phi_o2=phasors[1] if self.n > 1 else phasors[0],
            phi_o3=compensated[0],
        )

    def build_mv(self) -> MultipolarValue:
        raise NotImplementedError


class PseudoMultipolarCascade(BaseCascade):
    """Pseudo-multipolar stage where Σ may deviate from zero."""

    def build_mv(self) -> MultipolarValue:
        phasors = self._phasor * self.amplitude
        coeffs = {polarity: phasors[i] for i, polarity in enumerate(self.loka.polarities)}
        return MultipolarValue(self.loka, coeffs)


class PureMultipolarCascade(BaseCascade):
    """Final multipolar stage with Σ≈0 enforced via mutual coupling."""

    def build_mv(self) -> MultipolarValue:
        phasors = self._phasor * self.amplitude
        mean = np.mean(phasors)
        compensated = phasors - (1.0 - self.mutual_strength) * mean
        coeffs = {polarity: compensated[i] for i, polarity in enumerate(self.loka.polarities)}
        value = MultipolarValue(self.loka, coeffs)
        return value


__all__ = [
    "CascadeNodes",
    "PhasorCascade",
    "BaseCascade",
    "PseudoMultipolarCascade",
    "PureMultipolarCascade",
]
