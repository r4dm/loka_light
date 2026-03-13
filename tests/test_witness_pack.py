import json

import numpy as np

from loka_light import sigma_noise, witnesses
from loka_light.applications.scenarios import pseudo_quantum_witness_pack


def test_chsh_is_2sqrt2_for_reference_measurements() -> None:
    expected = 2.0 * np.sqrt(2.0)
    assert np.isclose(witnesses.chsh_value(), expected, atol=5e-3)


def test_cglmp_grows_with_dimension() -> None:
    values = [witnesses.cglmp_value(d) for d in (3, 4, 5)]
    assert values[0] > 0.0
    assert values[0] < values[1] < values[2]


def test_sigma_consistent_unitary_preserves_sigma_better_than_generic() -> None:
    dim = 10
    rng = np.random.default_rng(0)
    psi = (rng.normal(size=dim) + 1j * rng.normal(size=dim)).astype(np.complex128)
    psi = psi / float(np.linalg.norm(psi))
    sigma_in = complex(psi.sum())

    u_sigma = sigma_noise.unitary_sigma_consistent(dim, rng=np.random.default_rng(1), epsilon=0.5)
    u_generic = sigma_noise.unitary_generic(dim, rng=np.random.default_rng(2), epsilon=0.5)

    sigma_out_sigma = complex((u_sigma @ psi).sum())
    sigma_out_generic = complex((u_generic @ psi).sum())

    delta_sigma = float(abs(sigma_out_sigma - sigma_in))
    delta_generic = float(abs(sigma_out_generic - sigma_in))

    assert delta_sigma < 1e-10
    assert delta_sigma < delta_generic
    assert delta_generic > 1e-3


def test_probability_tensor_is_hermitian_rank_one_outer_product() -> None:
    from loka_light.physics.multipolar_wave import MultiConjugateFunction

    psi = np.array([1.0 + 0.0j, 0.5j, -0.3 + 0.1j], dtype=np.complex128)
    wave = MultiConjugateFunction(psi, n_conjugates=3)
    rho = wave.probability_tensor()

    assert np.allclose(rho, np.outer(psi, np.conj(psi)))
    assert np.allclose(rho, rho.conj().T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(rho)
    assert np.count_nonzero(eigvals > 1e-10) == 1
    assert np.all(eigvals >= -1e-10)


def test_pseudo_quantum_witness_pack_records_clean_state_before_noise(tmp_path) -> None:
    outdir = tmp_path / "witness_pack"
    pseudo_quantum_witness_pack({"outdir": str(outdir), "seed": 0, "noise_dim": 9, "d_values": [2, 3, 4]})

    summary = json.loads((outdir / "summary.json").read_text())
    sigma_noise_summary = summary["sigma_noise"]

    sigma_after_clean = complex(*sigma_noise_summary["sigma_after_clean"])
    sigma_out_sigma = complex(*sigma_noise_summary["sigma_out_sigma_consistent"])

    assert sigma_noise_summary["clean_state"] is True
    assert abs(sigma_after_clean) < 1e-10
    assert abs(sigma_out_sigma) < 1e-10
    assert sigma_noise_summary["delta_sigma_consistent"] < sigma_noise_summary["delta_sigma_generic"]
