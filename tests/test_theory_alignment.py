import numpy as np

from loka_light.core.algebras import LokaCn
from loka_light.core.value import MultipolarValue
from loka_light.devices.communication import MultipolarTransmitter
from loka_light.devices.components import MultiPlateCapacitor, NBranchInductor
from loka_light.devices.sources import MultipolarOscillator
from loka_light.physics.multipolar_wave import MultiConjugateFunction, WaveMetadata
from loka_light.physics.sigma import nx_stage, sigma_norm
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
