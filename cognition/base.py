"""Abstract mind contract ensuring every observer supplies its loka, transforms propositions into
Σ-balanced data, and reports configuration metadata for reproducible cascades."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence

from ..core.loka import Loka


class AbstractMind(ABC):
    """Interface that binds a mind to a concrete multipolar loka."""

    @abstractmethod
    def get_loka(self) -> Loka:
        """Return the loka whose tatva/dharma cascade this mind enforces."""

    @abstractmethod
    def analyse(self, propositions: Sequence[str]) -> Dict[str, Any]:
        """Convert propositions into Σ-balanced structures (weights or MV)."""

    @abstractmethod
    def to_metadata(self) -> Dict[str, Any]:
        """Expose configuration details for logs, runs, and verification."""
