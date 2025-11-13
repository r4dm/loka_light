AGENTS guide for loka_light (theory-first, ultra-brief)

This is a conceptual orientation for AI agents (not an API manual).

## Purpose (what to model)
- N‑polar information flow beyond binary (+/−)
- Minimal blocks: mind, oscillator, coder, receiver
- Reproducible demos linking messages → waves

## Core (what exists)
- Multipolarity = distribution over N polarity axes
- MultipolarOscillator shapes multi‑conjugate waves
- PolarCoder maps integers 0…N−1 to basis distributions
- Wave carries amplitudes + light metadata
- PseudoBlockM (devices/pseudomultipolar.py) sums multiple 2‑pole sources into C_n at O1 (relative ground)
- Sigma tools: `physics/sigma.py` → `p_perp`, `n_stage` (Σ→0), `nx_stage` (multi‑section)
- SigmaGuard device: `devices/sigma_guard.py` wraps N/NX for cascade O2/O3
- T‑composition: `physics/composition.py: compose_two_triads_to_c6()` (two triads → 6P)
- WaveMetadata carries `frequency_hz` for RX compatibility checks

## Operational model (how to think)
1) Pick N and output mode
2) Transmitter(+MultipolarOscillator) encodes messages → wave
3) Apply SigmaGuard (N or NX) where Σ→0 is required in the cascade (O2)
4) Focus on amplitude structure across polarities, not channels
5) Receiver checks compatibility and demodulates

## Pseudomultipolar vs Volumetric (quick map)
- Pseudomultipolar (network, M/N): `devices.pseudomultipolar.PseudoBlockM` (O1 summation, relative ground) and Σ‑projection utilities in `physics/sigma.py` (`p_perp`, `n_stage`, `nx_stage`) via `devices.sigma_guard.SigmaGuard`. Use them at O2/O3 to remove the common component and stabilize the differential signal across N and N₁…Nₓ. This is not a 3D field model.
- Volumetric (field): formation/radiation/reception in a medium via `devices.sources.MultipolarOscillator`, `devices.communication.MultipolarAntenna`, and `devices.detectors.MultipolarReceiver`. Selectivity is by number of poles and (optionally) frequency.
- Extras: `MultipolarOscillator` exposes a `geometry_profile` label; antennas support `loss_db` attenuation for simple channel modeling.
- Heuristic: for conversion/normalization and Σ control → SigmaGuard (M/N). For propagation/antennas/medium coupling → volumetric chain.

### Pseudo‑quantum (CPU) layer

- Location: `simulation.multipolar_pseudo_quantum` (CPU/NumPy only).
- State: reuses `physics.multipolar_wave.MultiConjugateFunction`.
- Scope: pseudo‑quantum M/N gates, circuits, Σ‑aware metrics and demos; no volumetric geometry.

## Minimal assumptions
- Messages in [0, N−1] (strict). No modulo unless explicit.
- Fixed N/frequency unless a dynamic key is used.
- Default demodulation = argmax → single index. For sequences, transmit stepwise.
- Demos aim to be deterministic and CI‑friendly.

## Heuristics (for agents)
- Reason over amplitude vectors, compare shapes
- Use `n_stage`/`nx_stage` to purify sums (Σ) before decoding
- For ordered exchange, send one message per transmission
- For pseudomultipolar demos, prefer `pseudo_mnx_chain` (M→NX→RX) to validate Σ trace

## Limits (by design)
- Minimal phenomenology; not a full EM simulator
- Default decode is argmax; advanced decoding is out‑of‑scope
- Theory lives in docstrings; no external PDFs are needed here
