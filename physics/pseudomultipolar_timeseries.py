"""NumPy time-series pseudomultipolar cascade (M → N → NX) with Σ diagnostics.

This module is a lightweight DSP/physics-toy "engine" for the network (M/N)
regime:

- O1: sources / M-style formation (a raw N-pole time series).
- O2: first N-stage (Σ removal) / first NX section.
- O3: last NX section (stabilised differential signal).

It deliberately reuses the same tap semantics as `physics.sigma.nx_stage`:
each section removes `tap` fraction of the mean component so that |Σ| decreases
monotonically across sections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

Array = np.ndarray


def generate_sources(
    n: int,
    *,
    steps: int = 256,
    carrier_cycles: float = 0.125,
    noise_std: float = 0.0,
    seed: int | None = 123,
) -> Array:
    """Generate an N-pole complex time series with a shared carrier and noise.

    The returned array has shape `(steps, n)` and is intended to represent the
    O1 node (raw pseudomultipolar formation output).
    """

    if n < 2:
        raise ValueError("n must be ≥ 2")
    if steps < 1:
        raise ValueError("steps must be ≥ 1")
    if carrier_cycles <= 0.0:
        raise ValueError("carrier_cycles must be > 0")
    if noise_std < 0.0:
        raise ValueError("noise_std must be ≥ 0")

    rng = np.random.default_rng(seed)
    base = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex128)
    if np.allclose(base, 0.0):
        base = (np.ones(n, dtype=np.complex128) + 0.0j) / np.sqrt(n)

    t = np.arange(steps, dtype=np.float64)
    carrier = np.exp(1j * 2.0 * np.pi * carrier_cycles * t / float(steps)).astype(np.complex128)
    series = carrier[:, None] * base[None, :]
    if noise_std > 0.0:
        series = series + noise_std * (
            rng.normal(size=(steps, n)).astype(np.float64) + 1j * rng.normal(size=(steps, n)).astype(np.float64)
        ).astype(np.complex128)
    return series


def default_profiles(*, sections: int = 3, tap: float = 0.5) -> List[float]:
    """Default NX tap profile (constant tap repeated `sections` times)."""

    if sections < 1:
        raise ValueError("sections must be ≥ 1")
    t = float(tap)
    if t <= 0.0 or t > 1.0:
        raise ValueError("tap values must satisfy 0 < tap ≤ 1")
    return [t] * int(sections)


def sigma_trace(signal: Array) -> Array:
    """Return per-sample |Σ| for a `(steps, n)` signal."""

    arr = np.asarray(signal, dtype=np.complex128)
    if arr.ndim != 2:
        raise ValueError("signal must have shape (steps, n)")
    return np.abs(arr.sum(axis=1)).astype(np.float64)


def energy_trace(signal: Array) -> Array:
    """Return per-sample energy Σ|x|^2 for a `(steps, n)` signal."""

    arr = np.asarray(signal, dtype=np.complex128)
    if arr.ndim != 2:
        raise ValueError("signal must have shape (steps, n)")
    return np.sum(np.abs(arr) ** 2, axis=1).astype(np.float64)


def project_rank(signal: Array, target_n: int) -> Array:
    """Project an `(steps, n)` signal down to `(steps, target_n)` by truncation.

    This intentionally models a *mismatched* receiver that cannot represent the
    full N-pole amplitude vector (e.g. n-1 poles). The projection is not meant
    to be a physical sensor model; it is a deterministic mismatch operator that
    makes RX energy/Σ diagnostics comparable in tests.
    """

    arr = np.asarray(signal, dtype=np.complex128)
    if arr.ndim != 2:
        raise ValueError("signal must have shape (steps, n)")
    if target_n < 2:
        raise ValueError("target_n must be ≥ 2")
    if target_n > arr.shape[1]:
        raise ValueError("target_n must be ≤ signal rank")
    return arr[:, :target_n].copy()


def _as_taps(sections: int | Sequence[float]) -> List[float]:
    if isinstance(sections, int):
        if sections < 1:
            raise ValueError("sections must be ≥ 1")
        return [1.0] * int(sections)
    taps = [float(x) for x in sections]
    if not taps:
        raise ValueError("sections must be ≥ 1")
    if any(tap <= 0.0 or tap > 1.0 for tap in taps):
        raise ValueError("tap values must satisfy 0 < tap ≤ 1")
    return taps


def run_cascade_multi(signal_o1: Array, *, sections: int | Sequence[float]) -> List[Array]:
    """Run NX sections on the time series and return each stage output."""

    arr = np.asarray(signal_o1, dtype=np.complex128)
    if arr.ndim != 2:
        raise ValueError("signal_o1 must have shape (steps, n)")
    taps = _as_taps(sections)
    steps, n = arr.shape
    ones = np.ones((1, n), dtype=np.complex128)

    out: List[Array] = []
    current = arr.copy()
    for tap in taps:
        sigma = current.sum(axis=1, keepdims=True)
        current = current - (tap / float(n)) * sigma * ones
        out.append(current.copy())
    return out


@dataclass(frozen=True)
class CascadeTimeseriesResult:
    """Time-series cascade output with Σ/energy telemetry."""

    taps: tuple[float, ...]
    o1: Array
    o2: Array
    o3: Array
    sigma_o1: Array
    sigma_chain: tuple[Array, ...]
    energy_o1: Array
    energy_o3: Array

    @property
    def mean_sigma_o1(self) -> float:
        return float(np.mean(self.sigma_o1))

    @property
    def mean_sigma_chain(self) -> List[float]:
        return [float(np.mean(trace)) for trace in self.sigma_chain]

    def is_sigma_monotone(self, *, atol: float = 1e-12) -> bool:
        """Return True if mean |Σ| decreases across NX sections."""

        means = self.mean_sigma_chain
        return all(means[i + 1] <= means[i] + atol for i in range(len(means) - 1))


def run_cascade(signal_o1: Array, *, sections: int | Sequence[float] = 1) -> CascadeTimeseriesResult:
    """Run the O1 → O2 → O3 cascade and return a compact result."""

    taps = tuple(_as_taps(sections))
    chain = run_cascade_multi(signal_o1, sections=taps)
    sigma_before = sigma_trace(signal_o1)
    sigma_chain = tuple(sigma_trace(stage) for stage in chain)
    return CascadeTimeseriesResult(
        taps=taps,
        o1=np.asarray(signal_o1, dtype=np.complex128),
        o2=chain[0],
        o3=chain[-1],
        sigma_o1=sigma_before,
        sigma_chain=sigma_chain,
        energy_o1=energy_trace(signal_o1),
        energy_o3=energy_trace(chain[-1]),
    )


__all__ = [
    "generate_sources",
    "default_profiles",
    "sigma_trace",
    "energy_trace",
    "project_rank",
    "run_cascade_multi",
    "CascadeTimeseriesResult",
    "run_cascade",
]

