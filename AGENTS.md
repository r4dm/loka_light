AGENTS guide for loka_light (theory-first, ultra-brief)

This is a conceptual orientation for AI agents (not an API manual).

## Purpose (what to model)
- N‑polar information flow beyond binary (+/−)
- Minimal blocks: loka algebra, mind, oscillator, coder, receiver
- Reproducible demos linking messages → waves → Σ‑cleanup → decode
- DSP/physics‑toy traces (time-series M→N→NX), plus small “passport” metrics (witnesses)

## Core (what exists)
- Multipolarity = distribution over N polarity axes
- Strict loka factory: `core/factory.py: create_loka()` (name normalization)
- Table-driven relational lokas: `core/algebras.py: GenericRelationalLoka` (incl. RelHexaSym/RelHeptaTPL3)
- MultipolarOscillator shapes multi‑conjugate waves
- PolarCoder maps integers 0…N−1 to basis distributions
- Wave carries amplitudes + metadata (`physics/multipolar_wave.py: WaveMetadata`, includes `frequency_hz`)
- PseudoBlockM (devices/pseudomultipolar.py) sums multiple 2‑pole sources into C_n at O1 (relative ground)
- Sigma tools: `physics/sigma.py` → `p_perp`, `sigma_residual`/`sigma_norm`, `n_stage` (Σ→0), `nx_stage` (multi‑section), with optional `linear_coeffs` to enforce weighted laws Σ_c = ∑(cᵢ·aᵢ)→0
- SigmaGuard device: `devices/sigma_guard.py` wraps N/NX for cascade O2/O3
- T‑composition: `physics/composition.py: compose_two_triads_to_c6()` (two triads → 6P)
- Shared-neutral superposition is the default in SuperpositionalLoka/Harloka (avoid accidental “two neutrals”)

## What’s new (reference points)
- v1.2.x: stricter algebra naming/creation, relational lokas, NX tap sections, consistent Σ conventions (Σ = linear sum), deterministic scenarios.
- v1.3.x: NumPy-first “validation pack” modules:
  - Time-series pseudomultipolar cascade: `physics/pseudomultipolar_timeseries.py`
  - Translation gap (n→2 projection artifact): `physics/translation_gap.py`
  - Witness pack (CHSH/CGLMP) + Σ-noise: `witnesses.py`, `sigma_noise.py`
- v1.4.x: weighted linear Σ laws via `linear_coeffs` (Σ_c = ∑(cᵢ·aᵢ)), complex Σ telemetry (`sigma_residual`), and theory-friendly conjugacy aliases on waves (`conjugacy_density`/`conjugacy_norm`).

## Operational model (how to think)
1) Pick N and output mode
2) Transmitter(+MultipolarOscillator) encodes messages → wave
3) Apply SigmaGuard (N or NX) where Σ→0 is required in the cascade (O2)
4) Focus on amplitude structure across polarities, not channels
5) Receiver checks compatibility and demodulates

## Pseudomultipolar vs Volumetric (quick map)
- Pseudomultipolar (network, M/N): `devices.pseudomultipolar.PseudoBlockM` (O1 summation, relative ground) and Σ‑projection utilities in `physics/sigma.py` (`p_perp`, `n_stage`, `nx_stage`, optional `linear_coeffs`) via `devices.sigma_guard.SigmaGuard`. Use them at O2/O3 to remove the common component (Σ or Σ_c) and stabilise the differential signal across N and N₁…Nₓ. This is not a 3D field model.
- Volumetric (field): formation/radiation/reception in a medium via `devices.sources.MultipolarOscillator`, `devices.communication.MultipolarAntenna`, and `devices.detectors.MultipolarReceiver`. Selectivity is by number of poles and (optionally) frequency.
- Extras: `MultipolarOscillator` exposes a `geometry_profile` label; antennas support `loss_db` attenuation for simple channel modeling.
- Heuristic: for conversion/normalization and Σ control → SigmaGuard (M/N). For propagation/antennas/medium coupling → volumetric chain.

### Pseudo‑quantum (CPU) layer

- Location: `simulation.multipolar_pseudo_quantum` (CPU/NumPy only).
- State: reuses `physics.multipolar_wave.MultiConjugateFunction`.
- Scalar metric: `MultiConjugateFunction.probability_density()` is defined as Σ|ψ|^k (k = `n_conjugates`).
- Alias: `conjugacy_density()` / `conjugacy_norm()` expose the same k‑metric in theory-friendly terms.
- Tensor metric: `MultiConjugateFunction.probability_tensor()` returns ρ = |ψ⟩⟨ψ|.
- Scope: pseudo‑quantum gates/circuits + witnesses/noise demos; no Torch, no volumetric loka geometry.

## Ready-to-run scenarios (entry points)
- `applications/scenarios.py: pseudo_mnx_chain` (M→NX→RX, Σ trace incl. complex residual; supports `linear_coeffs`)
- `applications/scenarios.py: pseudomultipolar_timeseries_demo` (DSP-style Σ/energy traces)
- `applications/scenarios.py: pseudo_quantum_witness_pack` (CHSH/CGLMP + Σ-noise vs generic)
- `applications/scenarios.py: translation_gap_demo` (n→2 projection loss/visibility)

## Minimal assumptions
- Messages in [0, N−1] (strict). No modulo unless explicit.
- Fixed N/frequency unless a dynamic key is used.
- Default demodulation = argmax → single index. For sequences, transmit stepwise.
- Demos aim to be deterministic and CI‑friendly.

## Heuristics (for agents)
- Reason over amplitude vectors, compare shapes
- Use `n_stage`/`nx_stage` to purify sums (Σ or Σ_c) before decoding; inspect `sigma_residual` when debugging
- For ordered exchange, send one message per transmission
- For pseudomultipolar demos, prefer `pseudo_mnx_chain` (M→NX→RX) to validate Σ trace

## Limits (by design)
- Minimal phenomenology; not a full EM simulator
- Default decode is argmax; advanced decoding is out‑of‑scope
- Theory lives in docstrings; no external PDFs are needed here
