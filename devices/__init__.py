"""Aggregates the source, detection, communication, media, and component classes for
convenient imports."""

from .sources import MultipolarOscillator, ElectrochemicalCell
from .detectors import MultipolarMicrophone, PolarityDetector, MultipolarReceiver
from .communication import MultipolarTransmitter, MultipolarAntenna, DynamicKey
from .components import (
    StackedCapacitor,
    MultiPlateCapacitor,
    NBranchInductor,
    TriCoil,
    TriCap,
)
from .media import MediaPhantom, MultipolarSpeaker, PhantomReproducer
from .coding import PolarCoder
from .passport import CascadeStage, StructuralPassport
from .pseudomultipolar import BipolarSource, PseudoBlockM

__all__ = [
    "MultipolarOscillator",
    "ElectrochemicalCell",
    "MultipolarMicrophone",
    "PolarityDetector",
    "MultipolarReceiver",
    "MultipolarTransmitter",
    "MultipolarAntenna",
    "DynamicKey",
    "StackedCapacitor",
    "MultiPlateCapacitor",
    "NBranchInductor",
    "TriCoil",
    "TriCap",
    "MediaPhantom",
    "PhantomReproducer",
    "MultipolarSpeaker",
    "PolarCoder",
    "CascadeStage",
    "StructuralPassport",
    "BipolarSource",
    "PseudoBlockM",
]
