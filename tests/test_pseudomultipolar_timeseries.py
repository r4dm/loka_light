import numpy as np

from loka_light.physics.pseudomultipolar_timeseries import (
    energy_trace,
    generate_sources,
    project_rank,
    run_cascade,
    sigma_trace,
)


def test_timeseries_n_stage_drives_mean_sigma_to_zero() -> None:
    sources = generate_sources(6, steps=128, seed=0)
    result = run_cascade(sources, sections=1)
    assert result.mean_sigma_chain[0] < 1e-10


def test_timeseries_nx_taps_decrease_mean_sigma_monotonically() -> None:
    sources = generate_sources(6, steps=128, seed=0)
    result = run_cascade(sources, sections=[0.5, 0.5, 0.5])

    means = result.mean_sigma_chain
    assert means[0] < result.mean_sigma_o1
    assert means[1] < means[0]
    assert means[2] < means[1]
    assert result.is_sigma_monotone()

    assert np.isclose(means[0], 0.5 * result.mean_sigma_o1, atol=1e-12)
    assert np.isclose(means[1], 0.25 * result.mean_sigma_o1, atol=1e-12)


def test_timeseries_linear_coeffs_drive_weighted_sigma_to_zero() -> None:
    sources = generate_sources(6, steps=128, seed=0)
    coeffs = [2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    result = run_cascade(sources, sections=1, linear_coeffs=coeffs)
    assert result.mean_sigma_chain[0] < 1e-10


def test_timeseries_linear_coeffs_taps_scale_mean_sigma() -> None:
    sources = generate_sources(6, steps=128, seed=0)
    coeffs = [2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    result = run_cascade(sources, sections=[0.5, 0.5], linear_coeffs=coeffs)
    means = result.mean_sigma_chain
    assert np.isclose(means[0], 0.5 * result.mean_sigma_o1, atol=1e-12)
    assert np.isclose(means[1], 0.25 * result.mean_sigma_o1, atol=1e-12)


def test_good_vs_bad_rx_is_distinguishable_by_energy() -> None:
    n = 6
    steps = 64
    base = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=np.complex128)
    carrier = np.exp(1j * 2.0 * np.pi * np.arange(steps, dtype=np.float64) / float(steps)).astype(np.complex128)
    sources = carrier[:, None] * base[None, :]

    result = run_cascade(sources, sections=1)
    good_energy = float(np.mean(energy_trace(result.o3)))
    bad = project_rank(result.o3, n - 1)
    bad_energy = float(np.mean(energy_trace(bad)))

    assert np.isclose(np.mean(sigma_trace(result.o3)), 0.0, atol=1e-12)
    assert bad_energy < good_energy
    assert np.isclose(good_energy, float(n), atol=1e-12)
    assert np.isclose(bad_energy, float(n - 1), atol=1e-12)
