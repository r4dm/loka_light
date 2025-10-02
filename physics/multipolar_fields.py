"""Minimal gauge-field abstractions for multipolar simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

from ..core.value import MultipolarValue


@dataclass
class GaugePoleField:
    """Spatially uniform multipolar field."""

    loka_name: str
    amplitudes: Tuple[complex, ...]

    def as_multipolar_value(self, loka) -> MultipolarValue:
        if len(loka.polarities) != len(self.amplitudes):
            raise ValueError("amplitude count must match loka rank")
        coeffs = {polarity: value for polarity, value in zip(loka.polarities, self.amplitudes)}
        return MultipolarValue(loka, coeffs)

    def energy(self) -> float:
        return float(np.sum(np.abs(self.amplitudes) ** 2))


@dataclass
class GaugePoleField2D:
    """Grid-based gauge field for quick visual experiments."""

    loka_name: str
    grid: np.ndarray

    def __post_init__(self) -> None:
        arr = np.asarray(self.grid, dtype=np.complex128)
        if arr.ndim != 3:
            raise ValueError("grid must be 3-D: height × width × poles")
        self.grid = arr

    def energy_map(self) -> np.ndarray:
        return np.sum(np.abs(self.grid) ** 2, axis=2)

    def snapshot(self, x: int, y: int) -> Tuple[complex, ...]:
        return tuple(self.grid[y, x])

    def iterate(self) -> Iterable[Tuple[int, int, Tuple[complex, ...]]]:
        h, w, _ = self.grid.shape
        for y in range(h):
            for x in range(w):
                yield x, y, self.snapshot(x, y)


__all__ = ["GaugePoleField", "GaugePoleField2D"]
