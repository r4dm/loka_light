"""Pseudomultipolar helpers: bipolar sources and block M mixer that sums them into an N-pole space.

This models the 'M' block as a network summation stage: several 2-pole sources
are mapped into a target N-pole loka, producing a composite `MultipolarValue`.
Downstream 'N' (SigmaGuard) is expected to perform Σ→0 purification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from ..core.algebras import LokaCn
from ..core.value import MultipolarValue
from .passport import CascadeStage, StructuralPassport


@dataclass
class BipolarSource:
    """Simple 2-pole source that emits a basis distribution for bit 0/1.

    Emits in a dedicated C2 (additive) loka as MultipolarValue. The mapping into
    the target N-pole space is performed by :class:`PseudoBlockM`.
    """

    name: str

    def loka(self) -> LokaCn:
        return LokaCn(2, operation_type="add", loka_name=f"{self.name}_C2", polarity_names=["n0", "n1"]) 

    def emit(self, bit: int) -> MultipolarValue:
        loc = self.loka()
        idx = 0 if int(bit) == 0 else 1
        coeffs = {loc.polarities[0]: 1.0 if idx == 0 else 0.0, loc.polarities[1]: 1.0 if idx == 1 else 0.0}
        return MultipolarValue(loc, coeffs)


class PseudoBlockM:
    """Mixer that sums multiple bipolar sources into an N-pole target loka.

    Parameters
    - target_n: number of poles in the resulting space (C_n, additive).
    - mapping: sequence assigning each source to a target polarity index 0..N-1.
    - weights: optional per-source linear weights (+/- for differential contribution).
    """

    def __init__(self, target_n: int, mapping: Sequence[int], *, name: str | None = None, weights: Sequence[float] | None = None) -> None:
        if target_n < 2:
            raise ValueError("target_n must be ≥ 2")
        self.name = name or "PseudoBlockM"
        self.target = LokaCn(target_n, operation_type="add", loka_name=f"C{target_n}", polarity_names=[f"P{i}" for i in range(target_n)])
        self.mapping = [int(i) for i in mapping]
        if any(i < 0 or i >= target_n for i in self.mapping):
            raise ValueError("mapping indices must be within 0..N-1")
        self.weights = [float(w) for w in (weights if weights is not None else [1.0] * len(self.mapping))]
        if len(self.weights) != len(self.mapping):
            raise ValueError("weights length must match mapping length")
        self.structural_passport = self._build_passport()

    def _build_passport(self) -> StructuralPassport:
        cascade = [
            CascadeStage(
                role="summation",
                loka=self.target.name,
                polarities=self.target.rank,
                description="Summation of multiple 2-pole sources into N-pole network node (O1)",
                operations=("bipolar→N mapping", "linear sum"),
                notes=("Acts as block M (pseudomultipolar)",),
            ),
        ]
        return StructuralPassport(
            device_name=self.name,
            cascade=cascade,
            nodes={"O1": "relative ground (summation node)"},
            ground_profiles=("relative_O1",),
            materials=("inductors/capacitors (abstract)",),
            notes=("Use SigmaGuard at O2/O3 for Σ→0 before decoding",),
        )

    def describe_structure(self) -> Dict[str, object]:
        return self.structural_passport.to_dict()

    def mix(self, sources: Sequence[BipolarSource], bits: Sequence[int]) -> MultipolarValue:
        if len(sources) != len(bits) or len(sources) != len(self.mapping):
            raise ValueError("sources, bits, and mapping must have the same length")
        coeffs = {pol: 0.0 for pol in self.target.polarities}
        for src, bit, idx, w in zip(sources, bits, self.mapping, self.weights):
            mv = src.emit(int(bit))
            # Interpret bipolar (1-of-2) as differential +1/-1 and sum into a single designated pole
            sign = 1.0 if mv.get_coefficient(mv.loka.polarities[0]).real > 0.5 else -1.0
            pol = self.target.polarities[idx]
            coeffs[pol] = coeffs.get(pol, 0.0) + w * sign
        return MultipolarValue(self.target, coeffs)


__all__ = ["BipolarSource", "PseudoBlockM"]

