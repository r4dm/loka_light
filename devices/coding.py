"""PolarCoder and DynamicKey utilities for mapping integers into multipolar coefficient spaces and
cycling through configured polarities and frequencies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from ..core.value import MultipolarValue
from ..core.loka import Loka
from ..cognition.base import AbstractMind
from .base import bind_mind_loka, MindLokaBinding


class PolarCoder:
    """Map integer messages to multipolar distributions."""

    def __init__(
        self,
        n: int | None = None,
        *,
        loka: MindLokaBinding | Loka | str | None = None,
        mind: AbstractMind | None = None,
    ) -> None:
        binding = loka if isinstance(loka, MindLokaBinding) else bind_mind_loka(mind=mind, loka=loka, n_hint=n)
        self._binding = binding
        self.n = binding.rank
        self._loka = binding.loka

    def rebind(
        self,
        *,
        n: int | None = None,
        loka: MindLokaBinding | Loka | str | None = None,
        mind: AbstractMind | None = None,
    ) -> None:
        binding = loka if isinstance(loka, MindLokaBinding) else bind_mind_loka(mind=mind, loka=loka, n_hint=n or self.n)
        self._binding = binding
        self.n = binding.rank
        self._loka = binding.loka

    @property
    def loka(self) -> Loka:
        return self._loka

    def encode(self, messages: Sequence[int], *, as_mv: bool = False) -> List[MultipolarValue | List[float]]:
        result: List[MultipolarValue | List[float]] = []
        for msg in messages:
            idx = int(msg)
            if not 0 <= idx < self.n:
                raise ValueError("Data out of range")
            weights = [0.0] * self.n
            weights[idx] = 1.0
            if as_mv:
                coeffs = {self._loka.polarities[i]: weights[i] for i in range(self.n)}
                result.append(MultipolarValue(self._loka, coeffs))
            else:
                result.append(weights)
        return result


@dataclass
class DynamicKey:
    """Cycle through predefined polarities/frequencies."""

    polarities: Sequence[int]
    freqs_hz: Sequence[float]

    def __post_init__(self) -> None:
        if not self.polarities:
            raise ValueError("polarities must not be empty")
        if any(p < 2 for p in self.polarities):
            raise ValueError("polarities must be ≥ 2")
        if not self.freqs_hz:
            raise ValueError("freqs_hz must not be empty")
        if any(f <= 0 for f in self.freqs_hz):
            raise ValueError("frequencies must be positive")
        if len(self.freqs_hz) not in (1, len(self.polarities)):
            raise ValueError("freqs_hz length must be 1 or match polarities length")
        if len(self.freqs_hz) == 1:
            self._freqs = [float(self.freqs_hz[0])] * len(self.polarities)
        else:
            self._freqs = [float(f) for f in self.freqs_hz]
        self._polarities = list(int(p) for p in self.polarities)
        self._idx = -1

    def next(self) -> tuple[int, float]:
        self._idx = (self._idx + 1) % len(self._polarities)
        return self._polarities[self._idx], self._freqs[self._idx]

    def peek(self) -> tuple[int, float]:
        next_idx = (self._idx + 1) % len(self._polarities)
        return self._polarities[next_idx], self._freqs[next_idx]

    def __iter__(self):  # pragma: no cover
        return self

    def __next__(self):  # pragma: no cover
        return self.next()


__all__ = ["PolarCoder", "DynamicKey"]
