"""Mind-aware helpers that resolve a consistent mind↔loka binding so instruments obey the Σ-balanced
laws activated by the observer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from ..core.loka import Loka
from ..core.factory import create_loka, normalize_loka_name
from ..cognition.base import AbstractMind


@dataclass(frozen=True)
class MindLokaBinding:
    """Resolve user input into a consistent (mind, loka) pairing."""

    mind: Optional[AbstractMind]
    loka: Loka

    @property
    def rank(self) -> int:
        return len(self.loka.polarities)


def bind_mind_loka(
    *,
    mind: Optional[AbstractMind] = None,
    loka: Union[Loka, str, None] = None,
    n_hint: Optional[int] = None,
) -> MindLokaBinding:
    """Return a binding that ensures devices share the same loka basis.

    Args:
        mind: Instance providing ``get_loka``.
        loka: Either a concrete :class:`Loka` or a name that can be resolved to
            a standard ``LokaCn``.  When ``mind`` is supplied it takes precedence.
        n_hint: Optional number of polarities to enforce when resolving a name.
    """

    if mind is not None:
        loka_obj = mind.get_loka()
        if not isinstance(loka_obj, Loka):
            raise TypeError("mind.get_loka() must return a Loka instance")
        return MindLokaBinding(mind=mind, loka=loka_obj)

    if isinstance(loka, Loka):
        return MindLokaBinding(mind=None, loka=loka)

    if isinstance(loka, str):
        loka_obj = create_loka(loka, n_hint=n_hint)
        return MindLokaBinding(mind=None, loka=loka_obj)

    if n_hint is not None and n_hint >= 2:
        name = f"AnonC{n_hint}"
        loka_obj = create_loka(name, n_hint=n_hint)
        return MindLokaBinding(mind=None, loka=loka_obj)

    raise ValueError(
        "mind or loka must be provided; supply a mind, loka object, loka name, or n_hint"
    )




class MindLinkedDevice:
    """Mixin that stores a shared mind/loka binding."""

    def __init__(
        self,
        *,
        mind: AbstractMind | None = None,
        loka: Loka | str | None = None,
        default_rank: int | None = None,
    ) -> None:
        self._binding = bind_mind_loka(mind=mind, loka=loka, n_hint=default_rank)

    @property
    def binding(self) -> MindLokaBinding:
        return self._binding

    @property
    def mind(self) -> AbstractMind | None:
        return self._binding.mind

    @property
    def loka(self) -> Loka:
        return self._binding.loka

    @property
    def rank(self) -> int:
        return self._binding.rank

    def rebind(
        self,
        *,
        mind: AbstractMind | None = None,
        loka: Loka | str | None = None,
        default_rank: int | None = None,
    ) -> None:
        self._binding = bind_mind_loka(mind=mind, loka=loka, n_hint=default_rank)

    def update_rank(self, n: int) -> None:
        if self._binding.mind is not None:
            raise ValueError("cannot change rank when device is bound to a mind")
        name = normalize_loka_name(self._binding.loka.name, n=int(n))
        self._binding = bind_mind_loka(loka=name, n_hint=int(n))


__all__ = ["MindLokaBinding", "bind_mind_loka", "MindLinkedDevice"]
