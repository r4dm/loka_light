"""Detection instruments—microphones, polarity scanners, receivers—that probe resonance and demodulate
multipolar signals for the active mind.

Note: in cascades, O2 ("Sigma guard") typically applies Σ→0 purification via
`devices.sigma_guard.SigmaGuard` using a single N-stage or a multi-section NX.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .coding import PolarCoder, DynamicKey
from .base import MindLinkedDevice
from ..core.loka import Loka
from .media import MediaPhantom
from .passport import CascadeStage, StructuralPassport
from .sources import MultipolarOscillator
from ..cognition.base import AbstractMind
from ..physics.multipolar_wave import MultiConjugateFunction


class MultipolarMicrophone:
    """Capture sound into pole-specific phase-shifted phantoms."""

    def __init__(self, polarity: int = 3, *, name: str | None = None) -> None:
        if polarity < 1:
            raise ValueError("polarity must be ≥ 1")
        self.polarity = polarity
        self.name = name or f"MultipolarMicrophone{polarity}P"

    def capture(self, source: MediaPhantom) -> List[MediaPhantom]:
        duration = source.duration_s
        sr = source.sample_rate
        freq = source.freq
        samples = int(duration * sr)
        t = np.arange(samples) / sr
        clones: List[MediaPhantom] = []
        for idx in range(self.polarity):
            phase = 2.0 * np.pi * idx / self.polarity
            buf = np.sin(2.0 * np.pi * freq * t + phase).astype(np.float32)
            phantom = MediaPhantom(
                duration_s=duration,
                sample_rate=sr,
                freq=freq,
                n_channels=1,
                intrinsic_polarity=idx,
                properties=dict(source.properties),
            )
            phantom.buffer = buf
            clones.append(phantom)
        return clones


@dataclass
class PolarityDetector:
    """Scan a phantom to find resonance at its intrinsic polarity."""

    phantom: MediaPhantom
    bandwidth: float = 1.5
    structural_passport: StructuralPassport = field(init=False)

    def __post_init__(self) -> None:
        self.structural_passport = self._build_passport()
        self._sync_passport()

    def measure_resonance(self, polarity: int) -> float:
        true_p = self.phantom.intrinsic_polarity
        if true_p is None:
            return 0.0
        response = float(np.exp(-((polarity - true_p) ** 2) / (2.0 * self.bandwidth ** 2)))
        self.structural_passport.record_metric("last_probe_poles", float(polarity))
        self.structural_passport.record_metric("last_response", response)
        return response

    def _build_passport(self) -> StructuralPassport:
        cascade = [
            CascadeStage(
                role="formation",
                loka="probe wave",
                polarities=max(self.phantom.intrinsic_polarity or 0, 2),
                description="Generate scanning wave and irradiate the phantom.",
            ),
            CascadeStage(
                role="measurement",
                loka="response",
                polarities=max(self.phantom.intrinsic_polarity or 0, 2),
                description="Evaluate resonance curve against expected polarity.",
            ),
        ]
        return StructuralPassport(
            device_name="PolarityDetector",
            cascade=cascade,
            nodes={"probe": "emission", "sensor": "response"},
            materials=("multipolar resonator", "phantom"),
            scenarios=("object_polarity_scan",),
        )

    def _sync_passport(self) -> None:
        intrinsic = self.phantom.intrinsic_polarity or 0
        n = intrinsic if intrinsic >= 2 else 2
        self.structural_passport.set_stage_polarities("formation", n)
        self.structural_passport.set_stage_polarities("measurement", n)

    def describe_structure(self) -> dict:
        self._sync_passport()
        return self.structural_passport.to_dict()


class MultipolarReceiver(MindLinkedDevice):
    """Receive multipolar waves and demodulate them into message indices.

    Cascade hint: O2 applies a SigmaGuard (N or NX) so the sum over poles is
    removed (Σ→0) before decoding.
    """

    def __init__(
        self,
        oscillator: MultipolarOscillator,
        *,
        key: DynamicKey | None = None,
        name: str | None = None,
        mind: AbstractMind | None = None,
        loka: Loka | str | None = None,
        polarity: int | None = None,
    ) -> None:
        desired_rank = polarity or oscillator.get_polarity()
        super().__init__(mind=mind, loka=loka, default_rank=desired_rank)
        self.oscillator = oscillator
        self.key = key
        self.name = name or "MultipolarReceiver"
        if self.oscillator.get_polarity() != self.rank:
            self.oscillator.set_polarity(self.rank)
        self._n_polarities = self.rank
        self._coder = PolarCoder(self._n_polarities, loka=self.loka)
        self._last_wave: Optional[MultiConjugateFunction] = None
        self.structural_passport = self._build_passport()
        self._sync_passport()

    def _build_passport(self) -> StructuralPassport:
        cascade = [
            CascadeStage(
                role="capture",
                loka="incoming wave",
                polarities=self._n_polarities,
                description="Antenna couples the wave and matches Sigma guard.",
            ),
            CascadeStage(
                role="demodulation",
                loka="information matrix",
                polarities=self._n_polarities,
                description="Polar coder converts amplitudes back into message indices.",
            ),
        ]
        notes = ["Receiver mirrors the transmitter's Sigma guard to close the cascade."]
        if self.key is not None:
            notes.append("Dynamic key retunes polarity and frequency before each capture.")
        return StructuralPassport(
            device_name=self.name,
            cascade=cascade,
            nodes={"O1": "antenna", "O2": "Sigma guard", "O3": "decoder"},
            materials=("Multipolar oscillator", "digital decoder"),
            notes=tuple(notes),
            ground_profiles=("relative_O2", "relative_O3"),
        )

    def _sync_passport(self) -> None:
        self._n_polarities = self.rank
        self.structural_passport.set_stage_polarities("capture", self._n_polarities)
        self.structural_passport.set_stage_polarities("demodulation", self._n_polarities)
        self.structural_passport.record_metric("configured_polarity", float(self._n_polarities))
        self.structural_passport.record_metric("working_frequency_hz", self.oscillator.working_frequency)

    def set_polarity(self, n_poles: int) -> None:
        if n_poles < 2:
            raise ValueError("n_poles must be ≥ 2")
        self.oscillator.set_polarity(n_poles)
        self.update_rank(int(n_poles))
        self._n_polarities = self.rank
        self._coder = PolarCoder(self._n_polarities, loka=self.loka)
        self._sync_passport()

    def is_compatible(self, wave: MultiConjugateFunction, *, tol_poles: int = 0, tol_freq: float | None = None) -> bool:
        poles_ok = abs(wave.n_conjugates - self.rank) <= tol_poles
        if not poles_ok:
            return False
        # Frequency check: only compare against explicit frequency metadata if requested
        if tol_freq is None or wave.metadata is None:
            return poles_ok
        freq_meta = getattr(wave.metadata, "frequency_hz", None)
        if freq_meta is None:
            return poles_ok
        return abs(freq_meta - self.oscillator.working_frequency) <= tol_freq

    def receive(self, wave: MultiConjugateFunction) -> bool:
        if self.key is not None:
            n_pol, freq = self.key.next()
            self.set_polarity(n_pol)
            self.oscillator.set_frequency(freq)
            self.structural_passport.record_metric("configured_polarity", float(self.rank))
        tol_freq = 1.0 if self.key is None else None
        if not self.is_compatible(wave, tol_freq=tol_freq):
            return False
        self._last_wave = wave
        return True

    def last_wave(self) -> Optional[MultiConjugateFunction]:
        return self._last_wave

    def demodulate(self, wave: Optional[MultiConjugateFunction] = None) -> List[int]:
        target = wave or self._last_wave
        if target is None:
            return []
        amps = target.amplitudes
        if amps.size == 0:
            return []
        idx = int(np.argmax(np.abs(amps)))
        self.structural_passport.record_metric("last_demodulated", float(idx))
        return [idx]

    def describe_structure(self) -> dict:
        self._sync_passport()
        return self.structural_passport.to_dict()


__all__ = [
    "MultipolarMicrophone",
    "PolarityDetector",
    "MultipolarReceiver",
]
