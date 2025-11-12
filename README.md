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

Pass keyword arguments through the dictionary literal to tweak behaviour; for
example, set `{"outdir": "runs/custom_scan", "true_polarity": 8}` when calling
`object_polarity_scan`.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{radaev2024loka,
  title={loka_light: Compact multipolar toolkit for n-polar information systems},
  author={Radaev, Maxim},
  year={2024},
  url={https://github.com/r4dm/loka_light},
  license={CC-BY-NC-4.0}
}
```

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
