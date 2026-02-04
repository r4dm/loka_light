import numpy as np

from loka_light import sigma_noise, witnesses


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
