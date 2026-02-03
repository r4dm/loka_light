"""Loka factory helpers.

This module provides a small name-based constructor for commonly used lokas.
It is intentionally conservative: if a name cannot be resolved, it raises.

The main purpose is to keep experiments reproducible by ensuring that a given
string resolves to the same loka definition everywhere in the codebase.
"""

from __future__ import annotations

import re
from typing import Dict, Optional

from .loka import Loka
from .algebras import GenericRelationalLoka, LokaCn

_CACHE: Dict[str, Loka] = {}


def normalize_loka_name(name: str, *, n: int) -> str:
    """Return a version of ``name`` whose first integer matches ``n``.

    If ``name`` contains no digits, append the rank as a suffix.
    """

    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")
    if n < 1:
        raise ValueError("n must be positive")
    match = re.search(r"(\d+)", name)
    if match is None:
        return f"{name}{n}"
    start, end = match.span(1)
    return f"{name[:start]}{n}{name[end:]}"


def _rel_hexa_sym() -> GenericRelationalLoka:
    # docs/mudrec.us.md/090_Шестиполярное_Пространство.md
    # a+b+c=β, d+δ+φ=β; a+δ=β, c+d=β, b+φ=β; full set → β
    rules: Dict[object, object] = {
        frozenset(["a", "b", "c"]): "β",
        frozenset(["d", "δ", "φ"]): "β",
        frozenset(["a", "δ"]): "β",
        frozenset(["c", "d"]): "β",
        frozenset(["b", "φ"]): "β",
        frozenset(["a", "b", "c", "d", "δ", "φ"]): "β",
    }
    return GenericRelationalLoka(
        loka_name="RelHexaSym",
        neutral_name="β",
        polarity_names=("a", "b", "c", "d", "δ", "φ"),
        rules=rules,
        tattva="relational_visibility",
        mind_modes=("hexa",),
    )


def _rel_hepta_tpl3() -> GenericRelationalLoka:
    # docs/mudrec.us.md/053_Cемиполярное_пространство.md and 090
    # Yantra 7: complementary pairs/triples → β, plus pair→polarity rules.
    rules: Dict[object, object] = {}
    for pair in [("A", "F"), ("B", "E"), ("C", "D")]:
        rules[frozenset(pair)] = "β"
    for tri in [("A", "B", "D"), ("C", "E", "F")]:
        rules[frozenset(tri)] = "β"
    pair_map = {
        ("A", "B"): "C",
        ("B", "D"): "F",
        ("A", "D"): "E",
        ("C", "F"): "B",
        ("C", "E"): "A",
        ("E", "F"): "D",
    }
    for (x, y), res in pair_map.items():
        rules[frozenset([x, y])] = res
    rules[frozenset(["A", "B", "C", "D", "E", "F"])] = "β"
    return GenericRelationalLoka(
        loka_name="RelHeptaTPL3",
        neutral_name="β",
        polarity_names=("A", "B", "C", "D", "E", "F"),
        rules=rules,
        tattva="relational_visibility",
        mind_modes=("hepta", "light"),
    )


_REL_RULESETS: Dict[str, callable] = {
    "RelHexaSym": _rel_hexa_sym,
    "RelHeptaTPL3": _rel_hepta_tpl3,
}


def create_loka(name: str, *, n_hint: Optional[int] = None) -> Loka:
    """Create a loka by name.

    Supported
    - Named relational lokas: ``RelHexaSym``, ``RelHeptaTPL3``.
    - Cyclic ``LokaCn`` inferred from the first integer in the name (or ``n_hint``).

    Notes
    - When ``n_hint`` is provided, the returned loka name is normalised so the
      embedded rank matches ``n_hint``. This avoids confusing artefacts where
      a loka name suggests one rank but the instance has another.
    """

    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")

    if name in _REL_RULESETS:
        if name not in _CACHE:
            _CACHE[name] = _REL_RULESETS[name]()
        loka = _CACHE[name]
        if n_hint is not None and int(n_hint) != loka.rank:
            raise ValueError(f"n_hint={n_hint} does not match fixed loka '{name}' (rank={loka.rank})")
        return loka

    match = re.search(r"(\d+)", name)
    n = int(match.group(1)) if match else None
    if n_hint is not None:
        n = int(n_hint)
        name = normalize_loka_name(name, n=n)
    if n is None or n < 2:
        raise ValueError("cannot infer loka rank from name; provide a numeric name or n_hint")

    if name in _CACHE:
        return _CACHE[name]

    operation_type = "add"
    if name.endswith("_mult"):
        operation_type = "multiply"
    elif name.endswith("_add"):
        operation_type = "add"

    polarity_names = [f"P{i}" for i in range(n)]
    loka = LokaCn(n=n, operation_type=operation_type, loka_name=name, polarity_names=polarity_names)
    _CACHE[name] = loka
    return loka


__all__ = ["create_loka", "normalize_loka_name"]
