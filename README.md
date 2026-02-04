[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17734819.svg)](https://doi.org/10.5281/zenodo.17734819)


# Loka Light

## Conceptual intro

- Multipolarity: signals are distributions over **N poles** instead of binary (+/−); `LokaCn` and `MultipolarValue` keep the algebra and Σ‑balance.
- Pseudomultipolar cascades: Σ is controlled by M/N/NX stages (`physics.sigma`, `devices.sigma_guard.SigmaGuard`) so that a common component is removed before decoding.
- Volumetric path (lightweight): `MultipolarOscillator` → TX/RX antennas → receiver form a simple medium/communication chain; `geometry_profile` is a label, not a full 3D field model.
- Pseudo‑quantum layer: CPU/NumPy states via `MultiConjugateFunction` with scalar `probability_density()` (= Σ|ψ|^k, k=`n_conjugates`) and tensor `probability_tensor()` for simple “quantum‑like” experiments.

### Minimal code examples

**1. Basic 4‑pole loka and Σ‑aware value**

```python
from loka_light.core.algebras import LokaCn
from loka_light.core.value import MultipolarValue

loka = LokaCn(4, "add", "C4_add", ["A", "B", "C", "D"])
mv = MultipolarValue(loka, {"A": 1.0, "C": -1.0})

print("mv:", mv)
print("collapsed:", mv.collapse())  # complex number with Σ structure
```

**2. SigmaGuard as a one‑line Σ→0 purification**

```python
from loka_light.core.algebras import LokaCn
from loka_light.core.value import MultipolarValue
from loka_light.devices.sigma_guard import SigmaGuard

loka = LokaCn(3, "add", "C3_add", ["P0", "P1", "P2"])
mv = MultipolarValue(loka, {"P0": 1.0, "P1": 0.5, "P2": -0.2})

guard = SigmaGuard()
mv_clean = guard.apply(mv)

print("residual before:", guard.residual(mv))
print("residual after:", guard.residual(mv_clean))
```

**3. Simple pseudo‑quantum state with tensor metric**

```python
import numpy as np
from loka_light.physics.multipolar_wave import MultiConjugateFunction

psi = MultiConjugateFunction([1.0 + 0.0j, 1.0j], n_conjugates=2)

print("probability_density:", psi.probability_density())
print("probability_tensor:\n", psi.probability_tensor())
```

## Installation

```bash
pip install loka-light
```

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Running Demo Scenarios with `python`

Each scenario is a plain function living under
`loka_light.applications.scenarios`.  Invoke them with
`python - <<'PY' ... PY` blocks (or adapt for your favourite runner).
The commands below write their artefacts into `runs/` subdirectories.

### 1. Object Polarity Scan

```bash
python - <<'PY'
from loka_light.applications.scenarios import object_polarity_scan
object_polarity_scan({})
PY
```

### 2. Secure Transmission Chain

```bash
python - <<'PY'
from loka_light.applications.scenarios import secure_transmission
secure_transmission({})
PY
```

Note on regimes: the secure chain uses the volumetric path (oscillator → TX/RX
antennas → receiver) for propagation in a medium, while Σ‑projection (M/N) is a
separate pseudomultipolar stage (see `devices/sigma_guard.py`) applied at O2/O3
to remove the common component before decoding.

### 3. Electrolyser Stage Update

```bash
python - <<'PY'
from loka_light.applications.scenarios import electrolyser_stage
electrolyser_stage({})
PY
```

### 4. Polarisation Field Split

```bash
python - <<'PY'
from loka_light.applications.scenarios import polarization_field
polarization_field({})
PY
```

### 5. Property Transfer Chain with Shared Mind

```bash
python - <<'PY'
from loka_light.applications.scenarios import property_transfer_chain
property_transfer_chain({})
PY
```

### 6. Structuring Field Application

```bash
python - <<'PY'
from loka_light.applications.scenarios import structuring_field
structuring_field({})
PY
```

### 7. Pseudo M→NX→RX Chain (Σ trace)

```bash
python - <<'PY'
from loka_light.applications.scenarios import pseudo_mnx_chain
pseudo_mnx_chain({"n": 6, "k": 3, "sections": 3, "bits": [1,0,1]})
PY
```

Writes `runs/pseudo_mnx_chain/trace.json` with the |Σ| values after each NX section
and the decoded index after Σ purification.

### 8. Pseudo‑Quantum H→Phase→H→Measure

```bash
python - <<'PY'
from loka_light.applications.scenarios import pseudo_quantum_hadamard_phase

pseudo_quantum_hadamard_phase({
    "shots": 256,
    "phase_angle": 1.0471975512,  # ~pi/3
    # "outdir": "runs/pseudo_quantum_hadamard",  # optional
})
PY
```

Writes `summary.json` under the chosen `outdir` with the measurement histogram,
Σ trace by stage and a snapshot of the final state.

## DSP/physics‑toy demos (NumPy)

These scenarios write machine‑readable artefacts (JSON/NPZ) under `runs/` so you
can inspect Σ traces, witness values, and projection losses without notebooks.

### 9. Pseudo‑Quantum Witness Pack (CHSH/CGLMP + Σ‑noise)

```bash
python - <<'PY'
from loka_light.applications.scenarios import pseudo_quantum_witness_pack
pseudo_quantum_witness_pack({"seed": 123, "epsilon": 0.2})
PY
```

Writes `runs/pseudo_quantum_witness_pack/summary.json` with CHSH/CGLMP values and
Σ‑invariant vs generic noise deltas (same seed/ε).

### 10. Pseudomultipolar Time‑Series Cascade (M→NX)

```bash
python - <<'PY'
from loka_light.applications.scenarios import pseudomultipolar_timeseries_demo
pseudomultipolar_timeseries_demo({"n": 6, "steps": 256, "sections": 3, "tap": 0.5, "seed": 123})
PY
```

Writes `runs/pseudomultipolar_timeseries/series.npz` (O1/O2/O3 signals + traces)
and `summary.json` (mean |Σ| per stage, monotonicity, RX mismatch metrics).

### 11. Translation Gap (n→2 projection)

```bash
python - <<'PY'
from loka_light.applications.scenarios import translation_gap_demo
translation_gap_demo({"n": 6, "seed": 123})
PY
```

Writes `runs/translation_gap/gap.npz` and `summary.json` with visibility/loss
metrics for a matched vs mismatched 2‑pole projection after Σ purification.

## Cascade Map (M → N/NX → RX)

```
  Pseudomultipolar (network)                       Volumetric (field)
  ┌──────────────┐   O1    ┌─────────┐   O2  ┌──────────┐  medium ┌──────────┐   O3  ┌──────────┐
  │  Block M     ├────────►│  N / NX │──────►│ TX ant.  │────────►│ RX ant.  │──────►│  Decoder │
  │ (sum 2‑pole) │  rel.G  │ Σ→0     │ rel.G │ gain+loss│         │ gain+loss│       │ (argmax) │
  └──────────────┘         └─────────┘       └──────────┘         └──────────┘       └──────────┘
```

- O1 (relative ground): summation node of `PseudoBlockM` (pseudomultipolar M‑stage).
- O2 (relative ground): `SigmaGuard` applies N or NX to drive Σ→0 before decode.
- TX/RX antennas: use `gain` and optional `loss_db` to model simple attenuation.
- Frequency: carried in `WaveMetadata.frequency_hz` and used by the receiver for compatibility.

Relevant APIs
- M‑stage: `devices.pseudomultipolar.PseudoBlockM`, `devices.pseudomultipolar.BipolarSource`.
- Σ‑stage: `physics.sigma` (P⊥/N/NX), `devices.sigma_guard.SigmaGuard`.
- Volumetric: `devices.sources.MultipolarOscillator` (with `geometry_profile` label),
  `devices.communication.MultipolarAntenna` (gain/loss), `devices.detectors.MultipolarReceiver`.

Pass keyword arguments through the dictionary literal to tweak behaviour; for
example, set `{"outdir": "runs/custom_scan", "true_polarity": 8}` when calling
`object_polarity_scan`.


## Direct Device Experiments

Sample notebook-free loop for the devices:

```bash
python - <<'PY'
from loka_light.cognition.models import NPoleMind
from loka_light.devices.sources import MultipolarOscillator, NBranchInductor, MultiPlateCapacitor
from loka_light.devices.communication import MultipolarTransmitter
from loka_light.devices.detectors import MultipolarReceiver

mind = NPoleMind(n=4, output_mode="mv")
inductor = NBranchInductor("L", ("n1", "n2"), n_branches=1, l_each=1e-3)
capacitor = MultiPlateCapacitor("C", ("n1", "n2"), n_plates=2, c_single=1e-6)

osc_tx = MultipolarOscillator([inductor], [capacitor], mind=mind)
osc_rx = MultipolarOscillator([inductor], [capacitor], mind=mind)

transmitter = MultipolarTransmitter(osc_tx, mind=mind)
receiver = MultipolarReceiver(osc_rx, mind=mind)

wave = transmitter.transmit([1, 2, 3])
receiver.receive(wave)
print(receiver.demodulate())
PY
```

## Pseudo‑Quantum CPU Simulator (multipolar)

The CPU implementation of the pseudo‑quantum simulator lives under
`loka_light.simulation.multipolar_pseudo_quantum` and works directly with
`MultiConjugateFunction` states.

Quick inline demo:

```bash
python - <<'PY'
import numpy as np

from loka_light.simulation import multipolar_pseudo_quantum as mpq

# Minimal H → phase → H → measure scenario
summary = mpq.hadamard_phase_measure_demo(shots=256, seed=42)
print("counts:", summary["counts"])

# Inspect correctness vs analytic reference
report = mpq.single_qubit_correctness_demo(phase_angle=np.pi/3, shots=1024, seed=123)
print("variation_distance:", report.variation_distance)
PY
```
