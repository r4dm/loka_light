"""Transmitter, antenna, and keying utilities that encode messages into multipolar waves and retune
polarity/frequency on demand.

Pseudomultipolar vs volumetric:
- Antenna/oscillator/receiver form the volumetric (field) chain — radiation and
  capture in a medium with selectivity by number of poles (and optionally frequency).
- SigmaGuard (N/NX) is a pseudomultipolar (network) stage placed at O2/O3 to
  project onto the Σ≈0 subspace before decoding. Do not conflate these roles:
  M/N ≠ radiation; antenna ≠ Σ‑projection.
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
        message_gain: float = 1.0,
        mind: AbstractMind | None = None,
        loka: Loka | str | None = None,
        polarity: int | None = None,
    ) -> None:
        desired_rank = polarity or oscillator.get_polarity()
        super().__init__(mind=mind, loka=loka, default_rank=desired_rank)
        self.oscillator = oscillator
        self.key = key
        self.message_gain = float(message_gain)
        if self.oscillator.get_polarity() != self.rank:
            self.oscillator.set_polarity(self.rank)
        # The oscillator defines the formation loka; keep the coder aligned to it.
        self._coder = PolarCoder(self.rank, loka=self.oscillator.loka)
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
                description="Multipolar oscillator shapes the outgoing wave.",
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
            self._coder = PolarCoder(self.rank, loka=self.oscillator.loka)
            self.structural_passport.record_metric("configured_polarity", float(self.rank))
        encoded = self._coder.encode(list(messages), as_mv=True)
        message_vec = np.zeros(self.rank, dtype=np.complex128)
        loka = self.oscillator.loka
        for mv in encoded:
            assert isinstance(mv, MultipolarValue)
            arr = np.array(
                [mv.coefficients.get(p, 0.0) for p in mv.loka.polarities], dtype=np.complex128
            )
            message_vec += arr
        if not encoded:
            raise ValueError("no messages provided")
        # Formation: oscillator produces the multipolar carrier (geometry + frequency).
        carrier = self.oscillator.generate_wave()
        if carrier.amplitudes.shape != message_vec.shape:
            raise ValueError("carrier and message vectors must have the same dimension")

        # Modulation: scale the formed carrier per pole using the message distribution.
        # This keeps phase geometry intact while making the targeted pole(s) dominant.
        out = carrier.amplitudes * (1.0 + (self.message_gain * message_vec))
        metadata = WaveMetadata.from_amplitudes(
            out,
            loka_name=loka.name,
            polarity_names=[p.name for p in loka.polarities],
            frequency_hz=self.oscillator.working_frequency,
        )
        wave = MultiConjugateFunction(out, n_conjugates=self.rank, metadata=metadata)
        self.structural_passport.record_metric("last_message_count", float(len(encoded)))
        return wave

    def describe_structure(self) -> Dict[str, object]:
        self._sync_passport()
        return self.structural_passport.to_dict()


class MultipolarAntenna:
    """Simple gain element with optional attenuation (loss) for tx/rx chains.

    Parameters
    - polarity: number of poles the antenna is tuned to.
    - role: informational label ("tx" or "rx").
    - gain: linear gain multiplier (applied after losses).
    - loss_db: additional attenuation in dB (applied symmetrically on emit/receive).
    """

    def __init__(self, *, polarity: int, role: str = "tx", gain: float = 1.0, loss_db: float = 0.0) -> None:
        self.polarity = int(polarity)
        self.role = role
        self.gain = float(gain)
        self.loss_db = float(loss_db)

    def _scale(self) -> float:
        # Convert dB loss to linear and apply gain
        loss_lin = 10.0 ** (-self.loss_db / 20.0) if self.loss_db != 0.0 else 1.0
        return self.gain * loss_lin

    def emit(self, wave: MultiConjugateFunction) -> MultiConjugateFunction:
        scaled = wave.copy()
        scaled.amplitudes = scaled.amplitudes * self._scale()
        if scaled.metadata is not None:
            scaled.metadata = WaveMetadata.from_amplitudes(
                scaled.amplitudes,
                loka_name=scaled.metadata.loka_name,
                polarity_names=scaled.metadata.polarity_names,
                frequency_hz=scaled.metadata.frequency_hz,
            )
        return scaled

    def receive(self, wave: MultiConjugateFunction) -> MultiConjugateFunction:
        scaled = wave.copy()
        scaled.amplitudes = scaled.amplitudes * self._scale()
        if scaled.metadata is not None:
            scaled.metadata = WaveMetadata.from_amplitudes(
                scaled.amplitudes,
                loka_name=scaled.metadata.loka_name,
                polarity_names=scaled.metadata.polarity_names,
                frequency_hz=scaled.metadata.frequency_hz,
            )
        return scaled


__all__ = ["MultipolarTransmitter", "MultipolarAntenna", "DynamicKey"]
