"""Minimal simulation harness tying minds, lokas, and devices together; it instantiates the chosen
loka, records structural passports, and offers Σ-aware analysis hooks."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from ..cognition.base import AbstractMind


class Engine:
    """Coordinate an `AbstractMind` with devices, scenarios, and metadata."""

    def __init__(self, mind: AbstractMind) -> None:
        self.mind = mind
        self._world: Optional[Dict[str, Any]] = None
        self._devices: Dict[str, Any] = {}

    def create_world(self) -> Dict[str, Any]:
        """Initialise world metadata using the mind's loka and configuration."""

        loka = self.mind.get_loka()
        self._world = {
            "loka": loka.to_metadata(),
            "mind": self.mind.to_metadata(),
            "devices": {},
        }
        return self._world

    def register_device(self, name: str, device: Any) -> None:
        """Attach a device and record its structural passport if available."""

        self._devices[name] = device
        if self._world is not None:
            self._world["devices"][name] = getattr(device, "describe_structure", lambda: None)()

    def run_analysis(self, propositions: Iterable[str]) -> Dict[str, Any]:
        """Evaluate propositions through the configured mind (Σ-aware output)."""

        return self.mind.analyse(list(propositions))

    def world_state(self) -> Dict[str, Any]:
        """Return cached world metadata, creating it on demand."""

        if self._world is None:
            return self.create_world()
        return self._world
