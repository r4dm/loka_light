"""Pseudo‑quantum simulator package built on multipolar waves.

Thesis:

- the state is stored as :class:`loka_light.physics.multipolar_wave.MultiConjugateFunction`;
- gates are linear (almost unitary) operations on amplitudes;
- M/N‑stages control Σ and implement pseudo‑multipolar removal;
- measurement interprets ``|ψ|^k`` (k = n_conjugates) as probabilities over polarities.

Implementation in this package:

- CPU primitives over ``MultiConjugateFunction`` (phase, unitary‑like
  matrices, M‑superpositions, N‑projections with Σ→0 monitoring);
- a lightweight DSL for "quantum‑like" circuits with telemetry on Σ and
  measurements;
- correctness metrics and CPU/NumPy benchmarks.

Important: this module stays in the pseudo‑multipolar cascade (M/N, Σ control)
and deliberately does not mix in volumetric device geometry. For waves
in media and antennas use `devices.sources`, `devices.communication`,
`devices.detectors` and other volumetric modules.
"""

from .gates import (
    apply_phase,
    apply_unitary,
    m_superpose,
    n_project,
    measure_polarity,
    sigma_value,
    sigma_residual,
    ensure_sigma_constraint,
)
from .circuit import (
    GateSpec,
    Circuit,
    CircuitTelemetry,
    run_circuit_cpu,
)
from .viz import state_summary, telemetry_summary
from .examples import (
    hadamard_phase_measure_demo,
    single_qubit_correctness_demo,
    performance_demo,
    batched_performance_demo,
    multi_conjugate_zero_sum_demo,
)
from .metrics import (
    variation_distance,
    CorrectnessReport,
    run_single_qubit_correctness_experiment,
    PerformancePoint,
    run_performance_benchmark,
    BatchPerformancePoint,
    run_batched_performance_benchmark,
)

__all__ = [
    "apply_phase",
    "apply_unitary",
    "m_superpose",
    "n_project",
    "measure_polarity",
    "sigma_value",
    "sigma_residual",
    "ensure_sigma_constraint",
    "GateSpec",
    "Circuit",
    "CircuitTelemetry",
    "run_circuit_cpu",
    "state_summary",
    "telemetry_summary",
    "hadamard_phase_measure_demo",
    "single_qubit_correctness_demo",
    "performance_demo",
    "batched_performance_demo",
    "multi_conjugate_zero_sum_demo",
    "variation_distance",
    "CorrectnessReport",
    "run_single_qubit_correctness_experiment",
    "PerformancePoint",
    "run_performance_benchmark",
    "BatchPerformancePoint",
    "run_batched_performance_benchmark",
]
