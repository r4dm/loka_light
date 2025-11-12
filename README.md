# Loka Light

## Environment

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
