"""Self-contained demo functions illustrating multipolar workflows such as polarity scans, secure
transmission, electrolyser stages, and field structuring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from ..core.algebras import LokaCn
from ..core.value import MultipolarValue
from ..devices import (
    MultipolarOscillator,
    MultipolarTransmitter,
    MultipolarReceiver,
    MultipolarAntenna,
    DynamicKey,
    MediaPhantom,
    PolarityDetector,
    MultipolarMicrophone,
    ElectrochemicalCell,
)
from ..physics.multipolar_wave import MultiConjugateFunction
from ..cognition.models import NPoleMind


def _outdir(params: Dict[str, Any], default: str) -> Path:
    out = Path(params.get("outdir", default))
    out.mkdir(parents=True, exist_ok=True)
    return out


def object_polarity_scan(params: Dict[str, Any]) -> None:
    true_polarity = int(params.get("true_polarity", 12))
    scan_min = int(params.get("n_min", 2))
    scan_max = int(params.get("n_max", 20))
    phantom = MediaPhantom(duration_s=0.5, intrinsic_polarity=true_polarity)
    detector = PolarityDetector(phantom)
    scan_range = list(range(scan_min, scan_max + 1))
    responses = [detector.measure_resonance(p) for p in scan_range]
    best = scan_range[int(np.argmax(responses))]
    outdir = _outdir(params, "runs/object_polarity_scan")
    np.save(outdir / "responses.npy", np.asarray(responses))
    (outdir / "summary.json").write_text(
        json.dumps({"true": true_polarity, "found": best}, indent=2)
    )


def secure_transmission(params: Dict[str, Any]) -> None:
    key = DynamicKey(polarities=[3, 4, 5, 6], freqs_hz=[110.0, 130.0, 150.0, 170.0])
    tx = MultipolarTransmitter(MultipolarOscillator([_default_inductor()], [_default_capacitor()]), key=key)
    rx = MultipolarReceiver(
        MultipolarOscillator([_default_inductor()], [_default_capacitor()]),
        key=DynamicKey(polarities=[3, 4, 5, 6], freqs_hz=[110.0, 130.0, 150.0, 170.0]),
    )
    tx_antennas: Dict[int, MultipolarAntenna] = {}
    rx_antennas: Dict[int, MultipolarAntenna] = {}
    messages = list(params.get("messages", [1, 0, 2, 3, 1, 3]))
    good: List[int] = []
    for msg in messages:
        wave = tx.transmit([msg])
        n = wave.n_conjugates
        tx_ant = tx_antennas.setdefault(n, MultipolarAntenna(polarity=n, role="tx", gain=1.05))
        rx_ant = rx_antennas.setdefault(n, MultipolarAntenna(polarity=n, role="rx", gain=0.95))
        emitted = tx_ant.emit(wave)
        received = rx_ant.receive(emitted)
        if rx.receive(received):
            good.extend(rx.demodulate())
    outdir = _outdir(params, "runs/secure_transmission")
    (outdir / "result.json").write_text(json.dumps({"sent": messages, "received": good}, indent=2))


def electrolyser_stage(params: Dict[str, Any]) -> None:
    phantom = MediaPhantom(duration_s=0.1, properties={"ph": 7.0})
    cell = ElectrochemicalCell(phantom, electrodes=["A", "B", "C"])
    currents = {"A": 0.2, "B": -0.2, "C": 0.0}
    cell.run_electrolysis(duration_s=0.02, currents=currents)
    outdir = _outdir(params, "runs/electrolyser_stage")
    (outdir / "properties.json").write_text(json.dumps(phantom.properties, indent=2))


def polarization_field(params: Dict[str, Any]) -> None:
    phantom = MediaPhantom(duration_s=0.1, properties={"density": 1.0})
    amps = np.random.default_rng(int(params.get("seed", 1))).random(6) + 0.1
    field = MultiConjugateFunction(amps, n_conjugates=6)
    fraction_a, fraction_b = phantom.apply_polarizing_field(field)
    outdir = _outdir(params, "runs/polarization_field")
    (outdir / "fraction_a.json").write_text(json.dumps(fraction_a.properties, indent=2))
    (outdir / "fraction_b.json").write_text(json.dumps(fraction_b.properties, indent=2))


def property_transfer_chain(params: Dict[str, Any]) -> None:
    phantom = MediaPhantom(duration_s=0.1, properties={"viscosity": 2.5, "conductivity": 0.8})
    mind = NPoleMind(n=4, output_mode="mv")
    messages = [int(max(0, min(3, round(v)))) for v in phantom.properties.values()]
    tx_osc = MultipolarOscillator([_default_inductor()], [_default_capacitor()], mind=mind)
    rx_osc = MultipolarOscillator([_default_inductor()], [_default_capacitor()], mind=mind)
    tx = MultipolarTransmitter(tx_osc, mind=mind)
    rx = MultipolarReceiver(rx_osc, mind=mind)
    decoded: List[int] = []
    for msg in messages:
        wave = tx.transmit([msg])
        rx.receive(wave)
        decoded.extend(rx.demodulate())
    outdir = _outdir(params, "runs/property_transfer")
    (outdir / "messages.json").write_text(json.dumps({"sent": messages, "decoded": decoded}, indent=2))


def structuring_field(params: Dict[str, Any]) -> None:
    phantom = MediaPhantom(duration_s=0.1, properties={"viscosity": 2.5, "octane": 70.0})
    amps = np.random.default_rng(int(params.get("seed", 2))).random(4) + 0.2
    field = MultiConjugateFunction(amps, n_conjugates=4)
    phantom.apply_structuring_field(field)
    outdir = _outdir(params, "runs/structuring_field")
    (outdir / "properties.json").write_text(json.dumps(phantom.properties, indent=2))


def _default_inductor() -> NBranchInductor:
    from ..devices.components import NBranchInductor

    return NBranchInductor("L", ("n1", "n2"), n_branches=1, l_each=1e-3)


def _default_capacitor() -> MultiPlateCapacitor:
    from ..devices.components import MultiPlateCapacitor

    return MultiPlateCapacitor("C", ("n1", "n2"), n_plates=2, c_single=1e-6)


__all__ = [
    "object_polarity_scan",
    "secure_transmission",
    "electrolyser_stage",
    "polarization_field",
    "property_transfer_chain",
    "structuring_field",
]
