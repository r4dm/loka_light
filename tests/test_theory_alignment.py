import pytest
import numpy as np

from loka_light.core.algebras import LokaCn
from loka_light.core.value import MultipolarValue
from loka_light.devices.communication import MultipolarAntenna, MultipolarTransmitter
from loka_light.devices.coding import DynamicKey
from loka_light.devices.components import MultiPlateCapacitor, NBranchInductor
from loka_light.devices.detectors import MultipolarReceiver
from loka_light.devices.sources import MultipolarOscillator
from loka_light.devices.sigma_guard import SigmaGuard
from loka_light.physics.multipolar_wave import MultiConjugateFunction, WaveMetadata
from loka_light.physics.sigma import n_stage, nx_stage, sigma_norm, sigma_residual
from loka_light.simulation.multipolar_pseudo_quantum.gates import measure_polarity


def test_wave_metadata_sigma_and_frequency() -> None:
    amps = np.array([1.0 + 0.0j, -1.0 + 0.0j], dtype=np.complex128)
    meta = WaveMetadata.from_amplitudes(
        amps,
        loka_name="X",
        polarity_names=["a", "b"],
        frequency_hz=100.0,
    )
    assert meta.sigma_residual == 0.0 + 0.0j
    assert meta.sigma_norm == 0.0
    assert meta.frequency_hz == 100.0


def test_multi_conjugate_normalize_uses_k_norm() -> None:
    psi2 = MultiConjugateFunction(np.array([1.0 + 0.0j, 1.0j]), n_conjugates=2)
    psi2.normalize()
    assert np.isclose(psi2.probability_density(), 1.0)

    psi4 = MultiConjugateFunction(np.array([1.0 + 0.0j, 1.0 + 0.0j]), n_conjugates=4)
    psi4.normalize()
    assert np.isclose(psi4.probability_density(), 1.0)


def test_measurement_uses_n_conjugate_power() -> None:
    state = MultiConjugateFunction(np.array([1.0 + 0.0j, 2.0 + 0.0j]), n_conjugates=4)
    _outcome, dist = measure_polarity(state, rng=np.random.default_rng(0))
    expected = np.array([1.0, 16.0]) / 17.0
    assert np.allclose(dist, expected)


def test_nx_stage_taps_decrease_sigma_monotonically() -> None:
    loka = LokaCn(3, operation_type="add", loka_name="C3", polarity_names=["A", "B", "C"])
    mv = MultipolarValue(loka, {loka.polarities[0]: 1.0})
    base = sigma_norm(mv)
    outs = nx_stage(mv, sections=[0.5, 0.5])
    residuals = [sigma_norm(out) for out in outs]
    assert residuals[0] < base
    assert residuals[1] < residuals[0]
    assert np.isclose(residuals[0], 0.5)
    assert np.isclose(residuals[1], 0.25)


def test_linear_form_n_stage_enforces_weighted_sigma() -> None:
    loka = LokaCn(3, operation_type="add", loka_name="C3", polarity_names=["A", "B", "C"])
    mv = MultipolarValue(loka, {loka.polarities[0]: 1.0})
    coeffs = [2.0, 1.0, 1.0]

    cleaned = n_stage(mv, linear_coeffs=coeffs)
    assert sigma_norm(cleaned, linear_coeffs=coeffs) < 1e-12
    assert abs(sigma_residual(cleaned, linear_coeffs=coeffs)) < 1e-12


def test_linear_form_nx_stage_reduces_residual_by_tap_factor() -> None:
    loka = LokaCn(3, operation_type="add", loka_name="C3", polarity_names=["A", "B", "C"])
    mv = MultipolarValue(loka, {loka.polarities[0]: 1.0})
    coeffs = [2.0, 1.0, 1.0]

    base = sigma_norm(mv, linear_coeffs=coeffs)
    outs = nx_stage(mv, sections=[0.5, 0.5], linear_coeffs=coeffs)
    residuals = [sigma_norm(out, linear_coeffs=coeffs) for out in outs]

    assert np.isclose(base, 2.0)
    assert np.isclose(residuals[0], 1.0)
    assert np.isclose(residuals[1], 0.5)


def test_conjugacy_aliases_match_probability_density() -> None:
    psi = MultiConjugateFunction(np.array([1.0 + 0.0j, 1.0 + 0.0j]), n_conjugates=4)
    assert np.isclose(psi.conjugacy_density(), psi.probability_density())
    assert np.isclose(psi.conjugacy_norm(), psi.probability_density() ** (1.0 / 4.0))
    assert np.isclose(psi.conjugacy_density(power=2), float(np.sum(np.abs(psi.amplitudes) ** 2)))


def test_probability_tensor_keeps_outer_product_structure() -> None:
    psi = np.array([1.0 + 0.0j, 0.5j, -0.25 + 0.0j], dtype=np.complex128)
    wave = MultiConjugateFunction(psi, n_conjugates=3)

    rho = wave.probability_tensor()

    assert np.allclose(rho, np.outer(psi, np.conj(psi)))
    assert np.allclose(rho, rho.conj().T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(rho) >= -1e-10)


def test_transmitter_modulates_oscillator_carrier() -> None:
    inductor = NBranchInductor("L", ("n1", "n2"), n_branches=1, l_each=1e-3)
    capacitor = MultiPlateCapacitor("C", ("n1", "n2"), n_plates=2, c_single=1e-6)
    osc = MultipolarOscillator([inductor], [capacitor], polarity=4)
    osc.set_frequency(123.0)

    carrier = osc.generate_wave()

    tx_carrier_only = MultipolarTransmitter(osc, message_gain=0.0)
    wave = tx_carrier_only.transmit([2])
    assert wave.metadata is not None
    assert wave.metadata.frequency_hz == 123.0
    assert np.allclose(wave.amplitudes, carrier.amplitudes)

    tx = MultipolarTransmitter(osc, message_gain=1.0)
    wave2 = tx.transmit([2])
    expected = carrier.amplitudes.copy()
    expected[2] = expected[2] * 2.0
    assert np.allclose(wave2.amplitudes, expected)


def test_antenna_rejects_pole_mismatch() -> None:
    antenna = MultipolarAntenna(polarity=4)
    wave = MultiConjugateFunction(np.ones(3, dtype=np.complex128), n_conjugates=3)

    with pytest.raises(ValueError):
        antenna.emit(wave)


def test_receiver_rejects_mismatched_metadata() -> None:
    inductor = NBranchInductor("L", ("n1", "n2"), n_branches=1, l_each=1e-3)
    capacitor = MultiPlateCapacitor("C", ("n1", "n2"), n_plates=2, c_single=1e-6)
    osc = MultipolarOscillator([inductor], [capacitor], polarity=4)
    osc.set_frequency(123.0)
    rx = MultipolarReceiver(osc, polarity=4)

    amps = np.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    wave = MultiConjugateFunction(
        amps,
        n_conjugates=4,
        metadata=WaveMetadata.from_amplitudes(
            amps,
            loka_name="WrongLoka",
            polarity_names=[p.name for p in rx.loka.polarities],
            frequency_hz=123.0,
        ),
    )

    assert not rx.receive(wave)


def test_receiver_keyed_mode_rejects_wrong_frequency() -> None:
    inductor = NBranchInductor("L", ("n1", "n2"), n_branches=1, l_each=1e-3)
    capacitor = MultiPlateCapacitor("C", ("n1", "n2"), n_plates=2, c_single=1e-6)
    osc = MultipolarOscillator([inductor], [capacitor], polarity=4)
    rx = MultipolarReceiver(
        osc,
        key=DynamicKey(polarities=[4], freqs_hz=[110.0]),
        polarity=4,
    )

    amps = np.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    wave = MultiConjugateFunction(
        amps,
        n_conjugates=4,
        metadata=WaveMetadata.from_amplitudes(
            amps,
            loka_name=rx.loka.name,
            polarity_names=[p.name for p in rx.loka.polarities],
            frequency_hz=999.0,
        ),
    )

    assert not rx.receive(wave)


def test_receiver_keyed_mode_requires_frequency_metadata() -> None:
    inductor = NBranchInductor("L", ("n1", "n2"), n_branches=1, l_each=1e-3)
    capacitor = MultiPlateCapacitor("C", ("n1", "n2"), n_plates=2, c_single=1e-6)
    osc = MultipolarOscillator([inductor], [capacitor], polarity=4)
    rx = MultipolarReceiver(
        osc,
        key=DynamicKey(polarities=[4], freqs_hz=[110.0]),
        polarity=4,
    )

    amps = np.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    wave_without_meta = MultiConjugateFunction(amps, n_conjugates=4)
    assert not rx.receive(wave_without_meta)

    wave_without_freq = MultiConjugateFunction(
        amps,
        n_conjugates=4,
        metadata=WaveMetadata.from_amplitudes(
            amps,
            loka_name=rx.loka.name,
            polarity_names=[p.name for p in rx.loka.polarities],
        ),
    )
    assert not rx.receive(wave_without_freq)


def test_receiver_purifies_before_demodulation_when_sigma_guard_is_active() -> None:
    inductor = NBranchInductor("L", ("n1", "n2"), n_branches=1, l_each=1e-3)
    capacitor = MultiPlateCapacitor("C", ("n1", "n2"), n_plates=2, c_single=1e-6)
    osc = MultipolarOscillator([inductor], [capacitor], polarity=4)
    osc.set_frequency(123.0)
    rx = MultipolarReceiver(osc, polarity=4, sigma_guard=SigmaGuard())

    amps = np.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    wave = MultiConjugateFunction(
        amps,
        n_conjugates=4,
        metadata=WaveMetadata.from_amplitudes(
            amps,
            loka_name=rx.loka.name,
            polarity_names=[p.name for p in rx.loka.polarities],
            frequency_hz=123.0,
        ),
    )

    assert rx.receive(wave)
    stored = rx.last_wave()
    assert stored is not None
    assert stored.metadata is not None
    assert stored.metadata.sigma_norm < 1e-12
    assert rx.demodulate() == [0]


def test_compose_two_triads_to_c6_is_constructible() -> None:
    from loka_light.physics.composition import compose_two_triads_to_c6

    t6, c3_left, c3_right = compose_two_triads_to_c6()
    assert t6.rank == 6
    assert [p.name for p in t6.polarities] == ["a", "b", "c", "A", "B", "C"]
    assert c3_left.rank == 3
    assert c3_right.rank == 3


def test_rel_hepta_tpl3_matches_yantra7_pairs_and_triples() -> None:
    from loka_light.core.factory import create_loka

    loka = create_loka("RelHeptaTPL3")
    beta = loka.neutral_element
    assert beta is not None

    A = loka.get_polarity_by_name("A")
    B = loka.get_polarity_by_name("B")
    C = loka.get_polarity_by_name("C")
    D = loka.get_polarity_by_name("D")
    E = loka.get_polarity_by_name("E")
    F = loka.get_polarity_by_name("F")
    assert all(p is not None for p in (A, B, C, D, E, F))

    assert loka.multiply(A, F) == beta
    assert loka.multiply(B, E) == beta
    assert loka.multiply(C, D) == beta

    assert loka.evaluate([A, B, D]) == beta
    assert loka.evaluate([C, E, F]) == beta

    assert loka.multiply(A, B) == C
    assert loka.multiply(B, D) == F
    assert loka.multiply(A, D) == E

    assert loka.inverse(A) == F
