"""Relativity helpers extending gamma factors and boosts to multipolar motion, including trihex
dynamics for three-pole cascades."""

from __future__ import annotations

import math
from typing import Tuple

C = 299_792_458


def gamma_n(v: float, n: int = 1) -> float:
    """Return the generalized gamma factor for n-pole motion."""

    if abs(v) >= C:
        raise ValueError("|v| must be smaller than c for real gamma")
    if n < 1:
        raise ValueError("n must be >= 1")
    return 1.0 / math.sqrt(1.0 - (v / C) ** (2 * n))


def transform_time(delta_t: float, v: float, n: int = 1) -> float:
    """Scale a time interval by the multipolar gamma factor."""

    return gamma_n(v, n) * delta_t


def transform_length(length: float, v: float, n: int = 1) -> float:
    """Contract a length using the multipolar gamma factor."""

    return length / gamma_n(v, n)


def _cubic_root(value: float) -> float:
    """Return a real cubic root that preserves the sign of the input."""

    return math.copysign(abs(value) ** (1.0 / 3.0), value)


def gamma_trihex(v: float) -> float:
    """Return the trihex gamma factor for three-pole cascades."""

    if abs(v) >= C:
        raise ValueError("|v| must be smaller than c for trihex gamma")
    beta3 = (v / C) ** 3
    if 1.0 + beta3 <= 0.0:
        raise ValueError("1 + (v/c)^3 must be positive for trihex gamma")
    return 1.0 / ((1.0 + beta3) ** (1.0 / 3.0))


def boost_trihex(delta_t: float, delta_x: float, v: float) -> Tuple[float, float]:
    """Apply the trihex boost preserving the cubic invariant (c·t)^3 - x^3."""

    beta3 = (v / C) ** 3
    denom = 1.0 + beta3
    if denom <= 0.0:
        raise ValueError("1 + (v/c)^3 must be positive for trihex boost")
    A = (C * delta_t) ** 3
    B = delta_x**3
    A_prime = (A - beta3 * B) / denom
    B_prime = (B - beta3 * A) / denom
    dt_prime = _cubic_root(A_prime) / C
    dx_prime = _cubic_root(B_prime)
    return dt_prime, dx_prime


__all__ = [
    "C",
    "gamma_n",
    "transform_time",
    "transform_length",
    "gamma_trihex",
    "boost_trihex",
]
