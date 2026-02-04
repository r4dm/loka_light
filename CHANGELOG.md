# Changelog

## 1.3.0 — 2026-02-04

- Added witness "passport" utilities: CHSH (d=2) and CGLMP (d>2) in `witnesses.py`.
- Added Σ-consistent vs generic unitary noise helpers in `sigma_noise.py`.
- Added NumPy time-series pseudomultipolar cascade engine (`physics/pseudomultipolar_timeseries.py`) and demo scenario writing `.npz` + `summary.json`.
- Added translation gap (n→2 projection) helpers (`physics/translation_gap.py`) and demo scenario writing `.npz` + `summary.json`.

## 1.2.0

- Sigma tools: `physics/sigma.py` (`p_perp`, `n_stage`, `nx_stage`) + `devices/sigma_guard.py`.
- CPU pseudo-quantum layer: `simulation/multipolar_pseudo_quantum`.
