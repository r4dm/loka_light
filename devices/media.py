"""Media abstractions for audio/visual experiments: phantom containers, phantom reproducer, and
multipolar speaker that recombine channelled waves."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, List

import numpy as np

from ..physics.multipolar_wave import MultiConjugateFunction


@dataclass
class MediaPhantom:
    """Simplified representation of a medium or object under observation."""

    duration_s: float
    sample_rate: float = 44_100.0
    freq: float = 440.0
    n_channels: int = 1
    intrinsic_polarity: Optional[int] = None
    properties: Dict[str, float] = field(default_factory=dict)

    buffer: Optional[np.ndarray] = field(default=None, repr=False)

    def clone(self) -> "MediaPhantom":
        copy = MediaPhantom(
            duration_s=self.duration_s,
            sample_rate=self.sample_rate,
            freq=self.freq,
            n_channels=self.n_channels,
            intrinsic_polarity=self.intrinsic_polarity,
            properties=dict(self.properties),
        )
        if self.buffer is not None:
            copy.buffer = self.buffer.copy()
        return copy

    def apply_structuring_field(self, field: MultiConjugateFunction) -> None:
        energy = field.probability_density()
        for key in list(self.properties):
            self.properties[key] = float(self.properties[key] + 0.01 * energy)

    def apply_polarizing_field(self, field: MultiConjugateFunction):
        clone_a = self.clone()
        clone_b = self.clone()
        energy = field.probability_density()
        clone_a.properties["polarized"] = float(energy)
        clone_b.properties["polarized"] = float(max(0.0, 1.0 - energy))
        return clone_a, clone_b


__all__ = ["MediaPhantom", "PhantomReproducer", "MultipolarSpeaker"]


class PhantomReproducer:
    """Sum multipolar audio channels into a single phantom waveform."""

    def __init__(self, weights: List[float] | None = None) -> None:
        self.weights = np.array(weights, dtype=float) if weights is not None else None

    def reproduce(self, channels: List[np.ndarray]) -> np.ndarray:
        if not channels:
            raise ValueError("channels must not be empty")
        lengths = {len(ch) for ch in channels}
        if len(lengths) != 1:
            raise ValueError("all channels must share the same length")
        weights = self.weights or np.ones(len(channels), dtype=float)
        if len(weights) != len(channels):
            raise ValueError("weights length must match channel count")
        stacked = np.stack(channels, axis=0)
        return (weights[:, None] * stacked).sum(axis=0)


class MultipolarSpeaker:
    """Recombine polar channels into a single playback buffer."""

    def __init__(self, *, weights: List[float] | None = None, name: str | None = None) -> None:
        self.reproducer = PhantomReproducer(weights)
        self.name = name or "MultipolarSpeaker"

    def play(self, channels: List[np.ndarray]) -> np.ndarray:
        return self.reproducer.reproduce(channels)
