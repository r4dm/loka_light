"""Polarity primitives representing the Σ-balanced units carried through M/N cascades; each stores
numeric projection, provenance, and depth so higher-level lokas can rebuild the ladder of
interactions."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .value import MultipolarValue


class Polarity:
    """Minimal unit of a multipolar cascade.

    Each polarity keeps a canonical name, an optional numeric value, and an
    optional reference to the `MultipolarValue` that generated it. Cascades
    use this information to keep Σ→0 balance across M/N layers while allowing
    derived poles to carry their depth level.
    """

    def __init__(
        self,
        name: str,
        value: complex | None = None,
        source_mv: Optional["MultipolarValue"] = None,
        level: Optional[int] = None,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("polarity name must be a non-empty string")
        if value is None and source_mv is not None:
            try:
                value = source_mv.collapse()
            except Exception:
                value = None

        self.name = name
        self.value: complex | None = value
        if level is None:
            self.level = (
                source_mv.level + 1
                if source_mv is not None and hasattr(source_mv, "level")
                else 0
            )
        else:
            self.level = level
        self.source_mv: Optional["MultipolarValue"] = source_mv

    def __repr__(self) -> str:
        if self.value is not None:
            if isinstance(self.value, complex):
                val_str = f"{self.value.real:.2f}{self.value.imag:+.2f}j"
            else:
                val_str = str(self.value)
            lvl = f", level={self.level}" if getattr(self, "level", 0) else ""
            return f"Polarity(name={self.name!r}{lvl}, value={val_str})"
        if self.source_mv is not None:
            lvl = f", level={self.level}" if getattr(self, "level", 0) else ""
            return f"Polarity(name={self.name!r}{lvl}, composite_from={self.source_mv})"
        lvl = f", level={self.level}" if getattr(self, "level", 0) else ""
        return f"Polarity(name={self.name!r}{lvl})"

    def __eq__(self, other) -> bool:
        if isinstance(other, Polarity):
            return self.name == other.name
        return False

    def __hash__(self) -> int:
        return hash(self.name)

    def numeric_value(self, recursive: bool = True):
        """Return a numeric projection of the polarity.

        When an explicit value is unavailable, the method optionally collapses
        the generating `MultipolarValue`, preserving Σ→0 conventions of the
        originating cascade.
        """

        if self.value is not None:
            return self.value
        if recursive and self.source_mv is not None:
            return self.source_mv.collapse(recursive=True)
        raise ValueError(f"Polarity '{self.name}' has no numeric value")
