"""Exports the polarity, loka, and value primitives that underpin the refactored multipolar core."""

from .polarity import Polarity
from .loka import DharmaProfile, TattvaProfile, Loka, register_dharma, register_tattva, get_dharma, get_tattva, list_dharmas, list_tattvas
from .value import MultipolarValue
from . import algebras

__all__ = [
    "Polarity",
    "DharmaProfile",
    "TattvaProfile",
    "Loka",
    "MultipolarValue",
    "register_dharma",
    "register_tattva",
    "get_dharma",
    "get_tattva",
    "list_dharmas",
    "list_tattvas",
    "algebras",
]
