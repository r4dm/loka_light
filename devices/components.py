"""Passive component models—multi-plate capacitors, bundled inductors, and triadic elements—used to
build multipolar formation stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class StackedCapacitor:
    """Equivalent capacitance for a plate stack."""

    name: str
    nodes: Tuple[str, str]
    n_plates: int
    c_single: float

    def __post_init__(self) -> None:
        if self.n_plates < 2:
            raise ValueError("n_plates must be ≥ 2")
        self.c_eq = self.c_single * (self.n_plates - 1)


class MultiPlateCapacitor(StackedCapacitor):
    """Alias kept for compatibility with older terminology."""


@dataclass
class NBranchInductor:
    """Parallel inductor bundle used by Multipolar oscillator."""

    name: str
    nodes: Tuple[str, str]
    n_branches: int
    l_each: float

    def __post_init__(self) -> None:
        if self.n_branches < 1:
            raise ValueError("n_branches must be ≥ 1")
        self.l_eq = self.l_each / self.n_branches


@dataclass
class TriCoil:
    """Three-coil series inductor."""

    name: str
    nodes: Tuple[str, str]
    l_single: float

    def __post_init__(self) -> None:
        self.l_eq = 3.0 * self.l_single


@dataclass
class TriCap:
    """Three-plate capacitor."""

    name: str
    nodes: Tuple[str, str]
    c_single: float

    def __post_init__(self) -> None:
        self.c_eq = 2.0 * self.c_single


__all__ = [
    "StackedCapacitor",
    "MultiPlateCapacitor",
    "NBranchInductor",
    "TriCoil",
    "TriCap",
]
