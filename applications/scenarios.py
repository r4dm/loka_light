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
from ..physics.multipolar_wave import MultiConjugateFunction, WaveMetadata
from ..cognition.models import NPoleMind
from ..devices.sigma_guard import SigmaGuard
from ..devices.pseudomultipolar import BipolarSource, PseudoBlockM
from ..physics.sigma import sigma_norm, sigma_residual


def _outdir(params: Dict[str, Any], default: str) -> Path:
    out = Path(params.get("outdir", default))
    out.mkdir(parents=True, exist_ok=True)
    return out


def _complex_dict(z: complex) -> Dict[str, float]:
    return {"re": float(z.real), "im": float(z.imag)}


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
            purified_vec = np.asarray(
                [purified.coefficients.get(p, 0.0) for p in purified.loka.polarities],
                dtype=np.complex128,
            )
            purified_wave = MultiConjugateFunction(
                purified,
                n_conjugates=received.n_conjugates,
                metadata=WaveMetadata.from_amplitudes(
                    purified_vec,
                    loka_name=purified.loka.name,
                    polarity_names=[p.name for p in purified.loka.polarities],
                    frequency_hz=(received.metadata.frequency_hz if received.metadata else None),
                ),
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
    "pseudomultipolar_timeseries_demo",
    "pseudo_quantum_witness_pack",
    "pseudo_quantum_hadamard_phase",
    "translation_gap_demo",
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
    linear_coeffs_raw = params.get("linear_coeffs")
    linear_coeffs = None if linear_coeffs_raw is None else [float(x) for x in linear_coeffs_raw]
    while len(bits) < k:
        bits.append(0)

    sources = [BipolarSource(f"S{i}") for i in range(k)]
    mapping = [i % n for i in range(k)]
    block_m = PseudoBlockM(n, mapping, name="BlockM")
    mv_mixed = block_m.mix(sources, bits)

    guard = SigmaGuard(sections=sections, linear_coeffs=linear_coeffs)
    sigma_before = sigma_norm(mv_mixed, linear_coeffs=linear_coeffs)
    sigma_before_residual = sigma_residual(mv_mixed, linear_coeffs=linear_coeffs)
    nx_values = guard.apply_nx(mv_mixed, sections=sections)
    sigma_trace = [sigma_norm(mv, linear_coeffs=linear_coeffs) for mv in nx_values]
    sigma_residual_trace = [sigma_residual(mv, linear_coeffs=linear_coeffs) for mv in nx_values]

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
                "linear_coeffs": linear_coeffs,
                "sigma_before": sigma_before,
                "sigma_before_residual": _complex_dict(sigma_before_residual),
                "sigma_trace": sigma_trace,
                "sigma_residual_trace": [_complex_dict(z) for z in sigma_residual_trace],
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
    seed = params.get("seed", 123)
    seed = None if seed is None else int(seed)
    outdir = _outdir(params, "runs/pseudo_quantum_hadamard")

    summary = mpq.hadamard_phase_measure_demo(
        shots=shots,
        phase_angle=phase_angle,
        seed=seed,
    )

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))


def pseudomultipolar_timeseries_demo(params: Dict[str, Any]) -> None:
    """Run a time-series M→NX cascade and persist arrays + Σ/energy summaries.

    Writes:
    - `series.npz` with O1/O2/O3 signals and Σ/energy traces.
    - `summary.json` with mean |Σ| (before and per NX section) and RX mismatch metrics.
    """

    from ..physics import pseudomultipolar_timeseries as pmts

    n = int(params.get("n", 6))
    steps = int(params.get("steps", 256))
    seed = params.get("seed", 123)
    seed = None if seed is None else int(seed)
    carrier_cycles = float(params.get("carrier_cycles", 0.125))
    noise_std = float(params.get("noise_std", 0.0))
    linear_coeffs_raw = params.get("linear_coeffs")
    linear_coeffs = None if linear_coeffs_raw is None else [float(x) for x in linear_coeffs_raw]

    taps: List[float]
    if "taps" in params:
        taps = [float(x) for x in params["taps"]]
    else:
        sections = int(params.get("sections", 3))
        tap = float(params.get("tap", 0.5))
        taps = pmts.default_profiles(sections=sections, tap=tap)

    sources = pmts.generate_sources(
        n,
        steps=steps,
        seed=seed,
        carrier_cycles=carrier_cycles,
        noise_std=noise_std,
    )
    result = pmts.run_cascade(sources, sections=taps, linear_coeffs=linear_coeffs)

    rx_bad_energy = None
    rx_bad_mean_sigma = None
    if n >= 3:
        bad = pmts.project_rank(result.o3, n - 1)
        rx_bad_energy = float(np.mean(pmts.energy_trace(bad)))
        bad_coeffs = None if linear_coeffs is None else linear_coeffs[: n - 1]
        rx_bad_mean_sigma = float(np.mean(pmts.sigma_trace(bad, linear_coeffs=bad_coeffs)))

    outdir = _outdir(params, "runs/pseudomultipolar_timeseries")
    np.savez(
        outdir / "series.npz",
        o1=result.o1,
        o2=result.o2,
        o3=result.o3,
        sigma_o1=result.sigma_o1,
        sigma_chain=np.stack(result.sigma_chain, axis=0),
        energy_o1=result.energy_o1,
        energy_o3=result.energy_o3,
    )
    (outdir / "summary.json").write_text(
        json.dumps(
            {
                "n": n,
                "steps": steps,
                "taps": list(result.taps),
                "seed": seed,
                "linear_coeffs": linear_coeffs,
                "mean_abs_sigma_o1": result.mean_sigma_o1,
                "mean_abs_sigma_chain": result.mean_sigma_chain,
                "sigma_monotone": result.is_sigma_monotone(),
                "rx_good_energy": float(np.mean(result.energy_o3)),
                "rx_bad_energy": rx_bad_energy,
                "rx_bad_mean_abs_sigma": rx_bad_mean_sigma,
            },
            indent=2,
        )
    )


def pseudo_quantum_witness_pack(params: Dict[str, Any]) -> None:
    """Compute CHSH/CGLMP reference values + Σ-consistent noise telemetry."""

    from .. import sigma_noise, witnesses

    seed = params.get("seed", 123)
    seed = None if seed is None else int(seed)
    epsilon = float(params.get("epsilon", 0.2))
    d_values = [int(x) for x in params.get("d_values", [2, 3, 4, 5])]
    d_values = [d for d in d_values if d >= 2]
    if not d_values:
        d_values = [2, 3, 4, 5]

    witness_table = {str(d): float(witnesses.cglmp_value(d)) for d in d_values}
    chsh = float(witnesses.chsh_value())

    dim = int(params.get("noise_dim", max(8, max(d_values) ** 2)))
    rng_state = np.random.default_rng(seed)
    psi = (rng_state.normal(size=dim) + 1j * rng_state.normal(size=dim)).astype(np.complex128)
    norm = float(np.linalg.norm(psi))
    if norm == 0.0:
        psi = (np.ones(dim, dtype=np.complex128) + 0.0j) / np.sqrt(dim)
    else:
        psi = psi / norm
    sigma_in = complex(psi.sum())

    rng_consistent = np.random.default_rng(None if seed is None else seed + 1)
    rng_generic = np.random.default_rng(None if seed is None else seed + 2)
    u_sigma = sigma_noise.unitary_sigma_consistent(dim, rng=rng_consistent, epsilon=epsilon)
    u_generic = sigma_noise.unitary_generic(dim, rng=rng_generic, epsilon=epsilon)
    sigma_out_sigma = complex((u_sigma @ psi).sum())
    sigma_out_generic = complex((u_generic @ psi).sum())

    outdir = _outdir(params, "runs/pseudo_quantum_witness_pack")
    (outdir / "summary.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "epsilon": epsilon,
                "witnesses": {
                    "chsh_d2": chsh,
                    "cglmp": witness_table,
                },
                "sigma_noise": {
                    "dim": dim,
                    "sigma_in": [float(sigma_in.real), float(sigma_in.imag)],
                    "sigma_out_sigma_consistent": [float(sigma_out_sigma.real), float(sigma_out_sigma.imag)],
                    "sigma_out_generic": [float(sigma_out_generic.real), float(sigma_out_generic.imag)],
                    "delta_sigma_consistent": float(abs(sigma_out_sigma - sigma_in)),
                    "delta_sigma_generic": float(abs(sigma_out_generic - sigma_in)),
                },
            },
            indent=2,
        )
    )


def translation_gap_demo(params: Dict[str, Any]) -> None:
    """Demonstrate the n→2 projection gap after Σ purification."""

    from ..physics.translation_gap import translation_gap

    n = int(params.get("n", 6))
    seed = params.get("seed", 123)
    seed = None if seed is None else int(seed)
    sigma_clean = bool(params.get("sigma_clean", True))

    result, x, p_match, p_mismatch, y_match, y_mismatch = translation_gap(
        n,
        seed=seed,
        sigma_clean=sigma_clean,
    )

    outdir = _outdir(params, "runs/translation_gap")
    np.savez(
        outdir / "gap.npz",
        x=x,
        p_match=p_match,
        p_mismatch=p_mismatch,
        y_match=y_match,
        y_mismatch=y_mismatch,
    )
    (outdir / "summary.json").write_text(
        json.dumps(
            {
                "n": result.n,
                "seed": seed,
                "sigma_clean": sigma_clean,
                "sigma_in": [float(result.sigma_in.real), float(result.sigma_in.imag)],
                "sigma_after_purify": [float(result.sigma_after_purify.real), float(result.sigma_after_purify.imag)],
                "visibility_match": float(result.visibility_match),
                "loss_match": float(result.loss_match),
                "visibility_mismatch": float(result.visibility_mismatch),
                "loss_mismatch": float(result.loss_mismatch),
            },
            indent=2,
        )
    )
