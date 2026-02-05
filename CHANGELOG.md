# Changelog

## 1.4.0 — 2026-02-05

- Added weighted linear Σ (Σ_c = ∑(cᵢ·aᵢ)) support via `linear_coeffs` in `physics/sigma.py` and `physics/pseudomultipolar_timeseries.py`.
- Added complex `sigma_residual()` (and extended `sigma_norm()`) for richer Σ diagnostics.
- Extended `devices/sigma_guard.SigmaGuard` to carry/apply `linear_coeffs`.
- Added theory-friendly k-metric aliases `conjugacy_density()` / `conjugacy_norm()` to `MultiConjugateFunction`.
- Updated demos and docs (README/AGENTS) to expose linear-form cleaning and residual traces.

## 1.3.0 — 2026-02-04

- Added witness "passport" utilities: CHSH (d=2) and CGLMP (d>2) in `witnesses.py`.
- Added Σ-consistent vs generic unitary noise helpers in `sigma_noise.py`.
- Added NumPy time-series pseudomultipolar cascade engine (`physics/pseudomultipolar_timeseries.py`) and demo scenario writing `.npz` + `summary.json`.
- Added translation gap (n→2 projection) helpers (`physics/translation_gap.py`) and demo scenario writing `.npz` + `summary.json`.

## 1.2.0

- Sigma tools: `physics/sigma.py` (`p_perp`, `n_stage`, `nx_stage`) + `devices/sigma_guard.py`.
- CPU pseudo-quantum layer: `simulation/multipolar_pseudo_quantum`.
