"""Source-side devices such as Multipolar oscillators and electrolytic cells that construct the formation
and release phases of a multipolar cascade."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

import numpy as np

from ..core.loka import Loka
from ..core.value import MultipolarValue
from ..cognition.base import AbstractMind
from .components import MultiPlateCapacitor, NBranchInductor
from .base import MindLinkedDevice
from .media import MediaPhantom
from .passport import CascadeStage, StructuralPassport
from ..physics.multipolar_wave import MultiConjugateFunction, WaveMetadata


class MultipolarOscillator(MindLinkedDevice):
    """Multipolar resonant circuit that keeps Σ→0 during formation."""

    def __init__(
        self,
        inductors: Sequence[NBranchInductor],
        capacitors: Sequence[MultiPlateCapacitor],
        *,
        name: str | None = None,
        mind: AbstractMind | None = None,
        loka: Loka | str | None = None,
        polarity: int | None = None,
        geometry_profile: str = "sphere",
    ) -> None:
        if not inductors or not capacitors:
            raise ValueError("oscillator requires at least one inductor and one capacitor")
        default_rank = polarity or 3
        super().__init__(mind=mind, loka=loka, default_rank=default_rank)
        self.name = name or "MultipolarOscillator"
        self.inductors = list(inductors)
        self.capacitors = list(capacitors)
        self._n_polarities = self.rank
        self._target_freq: float | None = None
        self.geometry_profile = str(geometry_profile)
        self.structural_passport = self._build_passport()
        self._sync_passport()

    # ------------------------------------------------------------------
    @property
    def L_eff(self) -> float:
        return sum(ind.l_eq for ind in self.inductors)

    @property
    def C_eff(self) -> float:
        return sum(cap.c_eq for cap in self.capacitors)

    def resonant_frequency(self) -> float:
        return 1.0 / (2.0 * math.pi * math.sqrt(self.L_eff * self.C_eff))

    @property
    def working_frequency(self) -> float:
        return self._target_freq if self._target_freq is not None else self.resonant_frequency()

    def set_frequency(self, freq_hz: float) -> None:
        if freq_hz <= 0:
            raise ValueError("frequency must be positive")
        self._target_freq = float(freq_hz)
        self._sync_passport()

    def set_polarity(self, n_poles: int) -> None:
        if n_poles < 2:
            raise ValueError("n_poles must be ≥ 2")
        self.update_rank(int(n_poles))
        self._n_polarities = self.rank
        self._sync_passport()

    def get_polarity(self) -> int:
        return self._n_polarities

    def generate_wave(self) -> MultiConjugateFunction:
        loka = self.loka
        polarities = list(loka.polarities)
        phases = np.exp(1j * 2.0 * np.pi * np.arange(self._n_polarities) / self._n_polarities)
        coeffs = {polarities[i]: phases[i] for i in range(self._n_polarities)}
        mv = MultipolarValue(loka, coeffs)
        metadata = WaveMetadata.from_amplitudes(
            phases,
            loka_name=loka.name,
            polarity_names=[p.name for p in loka.polarities],
            frequency_hz=self.working_frequency,
        )
        wave = MultiConjugateFunction(
            mv,
            n_conjugates=self._n_polarities,
            metadata=metadata,
        )
        return wave

    # ------------------------------------------------------------------
    def _build_passport(self) -> StructuralPassport:
        cascade = [
            CascadeStage(
                role="formation",
                loka=f"Cn_add",
                polarities=self._n_polarities,
                description="Inductor/Capacitor network accumulates energy and sets multipolar phase offsets.",
                operations=("inductor bundle", "capacitor stack"),
                notes=("Maintains Sigma balance during wave formation.",),
            ),
            CascadeStage(
                role="realization",
                loka="propagating wave",
                polarities=self._n_polarities,
                description="Wave is released into the medium or antenna.",
                operations=("coupling loop",),
                notes=("Prepared wave is ready for transmission.",),
            ),
        ]
        notes = ("Prepared wave is ready for transmission.", f"geometry_profile={self.geometry_profile}")
        return StructuralPassport(
            device_name=self.name,
            cascade=cascade,
            nodes={"O1": "energy intake", "O2": "Sigma guard", "O3": "output"},
            materials=("inductors", "capacitors"),
            notes=notes,
        )

    def _sync_passport(self) -> None:
        self.structural_passport.set_stage_polarities("formation", self._n_polarities)
        self.structural_passport.set_stage_polarities("realization", self._n_polarities)
        self.structural_passport.record_metric("configured_polarity", float(self._n_polarities))
        self.structural_passport.record_metric("working_frequency_hz", self.working_frequency)

    def describe_structure(self) -> Dict[str, object]:
        self._sync_passport()
        return self.structural_passport.to_dict()


class ElectrochemicalCell:
    """Simplified electrolyser that updates medium properties."""

    def __init__(self, phantom: MediaPhantom, electrodes: Iterable[str] | None = None, *, name: str | None = None) -> None:
        self.phantom = phantom
        self.electrodes = list(electrodes or ["A", "B", "C"])
        self.name = name or "ElectrochemicalCell"
        self.structural_passport = self._build_passport()
        self._sync_passport()

    def _build_passport(self) -> StructuralPassport:
        cascade = [
            CascadeStage(
                role="formation",
                loka="electrolytic field",
                polarities=max(len(self.electrodes), 2),
                description="Applied currents drive carrier production in the medium.",
            ),
            CascadeStage(
                role="realization",
                loka="structured medium",
                polarities=max(len(self.electrodes), 2),
                description="Updated properties are made available to downstream devices.",
            ),
        ]
        return StructuralPassport(
            device_name=self.name,
            cascade=cascade,
            nodes={elec: "electrode" for elec in self.electrodes},
            materials=("electrolyte", "electrodes"),
        )

    def _sync_passport(self) -> None:
        n = max(len(self.electrodes), 2)
        self.structural_passport.set_stage_polarities("formation", n)
        self.structural_passport.set_stage_polarities("realization", n)

    def run_electrolysis(self, duration_s: float, currents: Dict[str, float], *, k_prod: float = 1e-6) -> None:
        for electrode in self.electrodes:
            current = abs(float(currents.get(electrode, 0.0)))
            carriers = k_prod * current * duration_s
            key = f"carrier_{electrode}"
            self.phantom.properties[key] = self.phantom.properties.get(key, 0.0) + carriers
        self.structural_passport.record_metric("duration_s", float(duration_s))
        self.structural_passport.record_metric("total_charge_C", float(sum(abs(float(currents.get(e, 0.0))) for e in self.electrodes) * duration_s))
        self.structural_passport.record_metric("k_prod", float(k_prod))

    def describe_structure(self) -> Dict[str, object]:
        self._sync_passport()
        return self.structural_passport.to_dict()


__all__ = ["MultipolarOscillator", "ElectrochemicalCell"]
