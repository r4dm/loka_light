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
from ..devices.sigma_guard import SigmaGuard
from ..devices.pseudomultipolar import BipolarSource, PseudoBlockM
from ..physics.sigma import sigma_norm


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
    guard = SigmaGuard(sections=2)
    for msg in messages:
        wave = tx.transmit([msg])
        n = wave.n_conjugates
        tx_ant = tx_antennas.setdefault(n, MultipolarAntenna(polarity=n, role="tx", gain=1.05))
        rx_ant = rx_antennas.setdefault(n, MultipolarAntenna(polarity=n, role="rx", gain=0.95))
        emitted = tx_ant.emit(wave)
        received = rx_ant.receive(emitted)
        if rx.receive(received):
            # Apply SigmaGuard (NX) before decoding to enforce Σ→0
            mv = received.to_multipolar_value(rx.loka)
            purified = guard.apply_nx(mv, sections=guard.sections)[-1]
            purified_wave = MultiConjugateFunction(
                purified,
                n_conjugates=received.n_conjugates,
                metadata=received.metadata,
            )
            good.extend(rx.demodulate(purified_wave))
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
    "pseudo_mnx_chain",
    "pseudo_quantum_hadamard_phase",
]


def pseudo_mnx_chain(params: Dict[str, Any]) -> None:
    """Demonstrate M (summation of 2-pole sources) → N/NX → RX with |Σ| trace.

    Steps
    - Create k bipolar sources and map them into an N-pole space via block M.
    - Apply SigmaGuard NX and record sigma_norm after each section.
    - Build a wave from the final value and demodulate with a rank-N receiver.
    - Save the |Σ| trace and decoded index for inspection.
    """

    n = int(params.get("n", 6))
    k = int(params.get("k", 3))
    sections = int(params.get("sections", 3))
    bits = [int(x) & 1 for x in params.get("bits", [1, 0, 1])][:k]
    while len(bits) < k:
        bits.append(0)

    sources = [BipolarSource(f"S{i}") for i in range(k)]
    mapping = [i % n for i in range(k)]
    block_m = PseudoBlockM(n, mapping, name="BlockM")
    mv_mixed = block_m.mix(sources, bits)

    guard = SigmaGuard(sections=sections)
    sigma_before = sigma_norm(mv_mixed)
    nx_values = guard.apply_nx(mv_mixed, sections=sections)
    sigma_trace = [sigma_norm(mv) for mv in nx_values]

    # Build a minimal RX chain and demodulate
    tx_osc = MultipolarOscillator([_default_inductor()], [_default_capacitor()], polarity=n)
    rx_osc = MultipolarOscillator([_default_inductor()], [_default_capacitor()], polarity=n)
    rx = MultipolarReceiver(rx_osc, polarity=n)
    final_mv = nx_values[-1]
    wave = MultiConjugateFunction(final_mv, n_conjugates=n)
    rx.receive(wave)
    decoded = rx.demodulate(wave)

    outdir = _outdir(params, "runs/pseudo_mnx_chain")
    (outdir / "trace.json").write_text(
        json.dumps(
            {
                "n": n,
                "k": k,
                "bits": bits,
                "sigma_before": sigma_before,
                "sigma_trace": sigma_trace,
                "decoded": decoded,
                "block_m_passport": block_m.describe_structure(),
            },
            indent=2,
        )
    )


def pseudo_quantum_hadamard_phase(params: Dict[str, Any]) -> None:
    """Run the pseudo‑quantum H→phase→H→measure demo and persist artefacts.

    Parameters in ``params`` (all optional):
    - ``shots``: number of runs to collect the histogram;
    - ``phase_angle``: phase applied to the second pole (radians);
    - ``seed``: RNG seed for reproducibility;
    - ``outdir``: output directory (default ``runs/pseudo_quantum_hadamard``).

    The function writes ``summary.json`` with counts, Σ trace and a snapshot of
    the final state into the chosen directory.
    """

    from ..simulation import multipolar_pseudo_quantum as mpq

    shots = int(params.get("shots", 256))
    phase_angle = float(params.get("phase_angle", np.pi / 3.0))
    seed = params.get("seed")
    outdir = _outdir(params, "runs/pseudo_quantum_hadamard")

    summary = mpq.hadamard_phase_measure_demo(
        shots=shots,
        phase_angle=phase_angle,
        seed=seed,
    )

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
