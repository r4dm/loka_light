"""Implements NPoleMind—an observer that distributes attention across n poles, yields MultipolarValue
projections on demand, and tracks dominant focus for downstream devices."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from ..core.algebras import LokaCn
from ..core.value import MultipolarValue

from .base import AbstractMind


def _normalize(weights: List[float]) -> List[float]:
    """Return weights scaled so that Σ=1, guarding against a zero vector."""

    total = sum(weights)
    if total == 0:
        return [0.0 for _ in weights]
    return [w / total for w in weights]


@dataclass(slots=True)
class NPoleMind(AbstractMind):
    """n-pole mind that honours Sigma balance across all poles."""

    n: int = 4
    output_mode: str = "weights"
    _poles: List[str] = field(init=False)
    _loka: LokaCn = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n < 2:
            raise ValueError("n must be >= 2 for NPoleMind")
        self._poles = [f"P{i}" for i in range(self.n)]
        self._loka = LokaCn(
            n=self.n,
            operation_type="add",
            loka_name=f"MindC{self.n}",
            polarity_names=self._poles,
        )

    def get_loka(self) -> LokaCn:
        """Provide the loka that encodes the mind's polarity basis."""

        return self._loka

    def analyse(self, propositions: Sequence[str]) -> Dict[str, Any]:
        """Distribute each proposition across the pole space."""

        results: Dict[str, Any] = {}
        for proposition in propositions:
            weights = [0.0] * self.n
            for ch in proposition:
                idx = (ord(ch) * 1315423911) % self.n
                weights[idx] += 1.0
            weights = _normalize(weights)
            if self.output_mode == "mv":
                coeffs = {}
                for i in range(self.n):
                    polarity = self._loka.get_polarity_by_name(self._poles[i])
                    if polarity is None:
                        continue
                    coeffs[polarity] = weights[i]
                results[proposition] = MultipolarValue(self._loka, coeffs)
            else:
                results[proposition] = weights
        return results

    def dominant_pole(self, weights: Sequence[float]) -> str:
        """Return the pole with maximal weight, mirroring mind focus."""

        if len(weights) != self.n:
            raise ValueError("weights length must match n")
        idx = max(range(self.n), key=lambda i: weights[i])
        return self._poles[idx]

    def suggest_pole_space(self) -> str:
        """Expose a human-friendly label for reporting (e.g. "4-pole")."""

        return f"{self.n}-pole"

    def to_metadata(self) -> Dict[str, Any]:
        """Capture parameters for mind-route reports and reproducibility."""

        return {
            "loka": self._loka.name,
            "polarities": list(self._poles),
            "n": int(self.n),
            "output_mode": self.output_mode,
        }
