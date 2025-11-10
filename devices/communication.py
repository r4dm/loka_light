"""Transmitter, antenna, and keying utilities that encode messages into multipolar waves and retune
polarity/frequency on demand.

Note: in typical cascades, the O2 stage acts as a "Sigma guard" (Σ→0) using
`devices.sigma_guard.SigmaGuard` to apply N or NX before further processing.
"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from ..core.value import MultipolarValue
from ..core.loka import Loka
from ..cognition.base import AbstractMind
from .coding import PolarCoder, DynamicKey
from .base import MindLinkedDevice
from .passport import CascadeStage, StructuralPassport
from .sources import MultipolarOscillator
from ..physics.multipolar_wave import MultiConjugateFunction, WaveMetadata


class MultipolarTransmitter(MindLinkedDevice):
    """Encode messages and produce a multipolar wave using a multipolar oscillator."""

    def __init__(
        self,
        oscillator: MultipolarOscillator,
        *,
        key: DynamicKey | None = None,
        mind: AbstractMind | None = None,
        loka: Loka | str | None = None,
        polarity: int | None = None,
    ) -> None:
        desired_rank = polarity or oscillator.get_polarity()
        super().__init__(mind=mind, loka=loka, default_rank=desired_rank)
        self.oscillator = oscillator
        self.key = key
        if self.oscillator.get_polarity() != self.rank:
            self.oscillator.set_polarity(self.rank)
        self._coder = PolarCoder(self.rank, loka=self.loka)
        self.structural_passport = self._build_passport()
        self._sync_passport()

    def _build_passport(self) -> StructuralPassport:
        n = self.rank
        cascade = [
            CascadeStage(
                role="encoding",
                loka=f"messages_{n}p",
                polarities=n,
                description="Polar coder maps messages to multipolar distributions.",
            ),
            CascadeStage(
                role="formation",
                loka=f"C{n}",
                polarities=n,
                description="Lensky oscillator shapes the outgoing wave.",
            ),
        ]
        return StructuralPassport(
            device_name="MultipolarTransmitter",
            cascade=cascade,
            nodes={"O1": "mind input", "O2": "oscillator", "O3": "antenna"},
            materials=("digital coder", "multipolar oscillator"),
        )

    def _sync_passport(self) -> None:
        n = self.rank
        self.structural_passport.set_stage_polarities("encoding", n, loka=f"messages_{n}p")
        self.structural_passport.set_stage_polarities("formation", n, loka=f"C{n}")
        self.structural_passport.record_metric("configured_polarity", float(n))
        self.structural_passport.record_metric("working_frequency_hz", self.oscillator.working_frequency)

    def transmit(self, messages: Iterable[int]) -> MultiConjugateFunction:
        if self.key is not None:
            n_pol, freq = self.key.next()
            self.oscillator.set_polarity(n_pol)
            self.oscillator.set_frequency(freq)
            self.update_rank(n_pol)
            self._coder = PolarCoder(self.rank, loka=self.loka)
            self.structural_passport.record_metric("configured_polarity", float(self.rank))
        encoded = self._coder.encode(list(messages), as_mv=True)
        total = np.zeros(self.rank, dtype=np.complex128)
        loka = None
        for mv in encoded:
            assert isinstance(mv, MultipolarValue)
            loka = mv.loka
            arr = np.array([mv.coefficients.get(p, 0.0) for p in mv.loka.polarities], dtype=np.complex128)
            total += arr
        if loka is None:
            raise ValueError("no messages provided")
        coeffs = {polarity: total[i] for i, polarity in enumerate(loka.polarities)}
        mv = MultipolarValue(loka, coeffs)
        wave = MultiConjugateFunction(
            mv,
            n_conjugates=self.rank,
            metadata=WaveMetadata(
                loka_name=loka.name,
                polarity_names=[p.name for p in loka.polarities],
                sigma_norm=float(np.linalg.norm(total)),
                sigma_residual=mv.collapse(),
            ),
        )
        if wave.metadata is not None:
            setattr(wave.metadata, 'frequency_hz', self.oscillator.working_frequency)
        self.structural_passport.record_metric("last_message_count", float(len(encoded)))
        return wave

    def describe_structure(self) -> Dict[str, object]:
        self._sync_passport()
        return self.structural_passport.to_dict()


class MultipolarAntenna:
    """Simple gain element used on both transmission and reception side."""

    def __init__(self, *, polarity: int, role: str = "tx", gain: float = 1.0) -> None:
        self.polarity = int(polarity)
        self.role = role
        self.gain = float(gain)

    def emit(self, wave: MultiConjugateFunction) -> MultiConjugateFunction:
        scaled = wave.copy()
        scaled.amplitudes = scaled.amplitudes * self.gain
        return scaled

    def receive(self, wave: MultiConjugateFunction) -> MultiConjugateFunction:
        scaled = wave.copy()
        scaled.amplitudes = scaled.amplitudes * self.gain
        return scaled


__all__ = ["MultipolarTransmitter", "MultipolarAntenna", "DynamicKey"]
