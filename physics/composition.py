"""Minimal composition helpers (T-composition: two triads → 6P).

This module provides a single factory that builds a 6-pole space from two
independent triads and wires them into one superpositional loka. The pairing
convention is a/A, b/B, c/C so downstream code can treat each pair as
conjugate/complimentary in device logic.
"""

from __future__ import annotations

from typing import Tuple

from ..core.algebras import LokaCn, SuperpositionLayer, SuperpositionalLoka


def compose_two_triads_to_c6(name: str = "T6") -> Tuple[SuperpositionalLoka, LokaCn, LokaCn]:
    """Return a (C6, C3_left, C3_right) superposition for triad→6P composition.

    - Resulting loka (C6) has polarities [a,b,c,A,B,C].
    - Left triad maps a→a, b→b, c→c.
    - Right triad maps A→A, B→B, C→C.

    The function keeps names explicit and avoids external conventions so models
    can reason over pairs (a/A, b/B, c/C) when building 6P receivers or guards.
    """

    c3_left = LokaCn(3, operation_type="add", loka_name="C3_left", polarity_names=["a", "b", "c"])
    c3_right = LokaCn(3, operation_type="add", loka_name="C3_right", polarity_names=["A", "B", "C"])
    c6 = LokaCn(6, operation_type="add", loka_name=name, polarity_names=["a", "b", "c", "A", "B", "C"])

    layer_left = SuperpositionLayer(
        name="left",
        source_loka=c3_left,
        mapping={"a": "a", "b": "b", "c": "c"},
        description="Left triad projected into the first three poles of C6",
    )
    layer_right = SuperpositionLayer(
        name="right",
        source_loka=c3_right,
        mapping={"A": "A", "B": "B", "C": "C"},
        description="Right triad projected into the last three poles of C6",
    )

    t6 = SuperpositionalLoka(
        name=f"{name}_superposed",
        polarities=list(c6.polarities),
        layers=(layer_left, layer_right),
        delegate_layer=layer_left,
        tattva="triadic_superposition",
        mind_modes=("triadic", "tetradic"),
    )
    return t6, c3_left, c3_right

