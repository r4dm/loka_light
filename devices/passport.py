"""Structural passport helpers capturing cascade stages, polarity counts, nodes, and metrics for each
instrument."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence


@dataclass
class CascadeStage:
    """Single stage of a device cascade."""

    role: str
    loka: str
    polarities: int
    description: str
    operations: Sequence[str] = field(default_factory=tuple)
    notes: Sequence[str] = field(default_factory=tuple)


@dataclass
class StructuralPassport:
    """Compact record describing how a device maintains Sigma→0."""

    device_name: str
    cascade: List[CascadeStage]
    nodes: Dict[str, str] = field(default_factory=dict)
    materials: Sequence[str] = field(default_factory=tuple)
    scenarios: Sequence[str] = field(default_factory=tuple)
    mind_interfaces: Sequence[str] = field(default_factory=tuple)
    ground_profiles: Sequence[str] = field(default_factory=tuple)
    notes: Sequence[str] = field(default_factory=tuple)

    metrics: Dict[str, float] = field(default_factory=dict)

    def set_stage_polarities(self, role: str, polarities: int, *, loka: str | None = None) -> None:
        for stage in self.cascade:
            if stage.role == role:
                stage.polarities = polarities
                if loka is not None:
                    stage.loka = loka
                return
        raise KeyError(f"stage '{role}' not found")

    def record_metric(self, name: str, value: float) -> None:
        self.metrics[name] = float(value)

    def to_dict(self) -> Dict[str, object]:
        return {
            "device": self.device_name,
            "cascade": [
                {
                    "role": stage.role,
                    "loka": stage.loka,
                    "polarities": stage.polarities,
                    "description": stage.description,
                    "operations": list(stage.operations),
                    "notes": list(stage.notes),
                }
                for stage in self.cascade
            ],
            "nodes": dict(self.nodes),
            "materials": list(self.materials),
            "scenarios": list(self.scenarios),
            "mind_interfaces": list(self.mind_interfaces),
            "ground_profiles": list(self.ground_profiles),
            "notes": list(self.notes),
            "metrics": dict(self.metrics),
        }

__all__ = ["CascadeStage", "StructuralPassport"]
