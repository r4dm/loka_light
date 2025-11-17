"""Usage examples for the pseudo‑quantum simulator.

The functions here are lightweight helpers suitable for notebooks or CLI
wrappers to generate artefacts and sanity‑check behaviour.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ...physics.multipolar_wave import MultiConjugateFunction

from .circuit import Circuit, GateSpec
from .viz import state_summary, telemetry_summary
from .gates import apply_unitary, n_project, sigma_value
from .metrics import (
    CorrectnessReport,
    PerformancePoint,
    BatchPerformancePoint,
    run_single_qubit_correctness_experiment,
    run_performance_benchmark,
    run_batched_performance_benchmark,
)


def hadamard_phase_measure_demo(
    *,
    shots: int = 256,
    phase_angle: float = np.pi / 3.0,
    seed: int | None = None,
) -> Dict[str, Any]:
    """Minimal scenario: H → phase → H → measure.

    Returns a dict with the measurement histogram, sample Σ trace by stage and
    a snapshot of the final state (for a single run).
    """

    rng = np.random.default_rng(seed)

    # Base state |0>
    psi0 = MultiConjugateFunction(np.array([1.0 + 0.0j, 0.0 + 0.0j]), n_conjugates=2)

    # Hadamard matrix analog
    H = (1.0 / np.sqrt(2.0)) * np.array(
        [[1.0, 1.0], [1.0, -1.0]],
        dtype=np.complex128,
    )

    circuit = Circuit(
        stages=[
            GateSpec(kind="unitary", params={"matrix": H}),
            # Phase shift for the second pole only
            GateSpec(kind="phase", params={"angle": np.array([0.0, phase_angle], dtype=float)}),
            GateSpec(kind="unitary", params={"matrix": H}),
            GateSpec(kind="measure", params={}),
        ]
    )

    counts = {0: 0, 1: 0}
    example_telemetry = None
    example_final = None

    from .circuit import run_circuit_cpu

    for shot in range(shots):
        final_state, telemetry = run_circuit_cpu(psi0, circuit, rng=rng)
        stage_idx, outcome, _dist = telemetry.measurements[-1]
        counts[outcome] = counts.get(outcome, 0) + 1
        if shot == 0:
            example_telemetry = telemetry
            example_final = final_state

    assert example_telemetry is not None and example_final is not None

    return {
        "shots": shots,
        "phase_angle": float(phase_angle),
        "counts": {int(k): int(v) for k, v in counts.items()},
        "example_sigma_trace": telemetry_summary(example_telemetry),
        "example_final_state": state_summary(example_final),
    }


def single_qubit_correctness_demo(
    *, phase_angle: float = np.pi / 3.0, shots: int = 1024, seed: int | None = 123
) -> CorrectnessReport:
    """Wrapper around the single‑qubit correctness experiment.

    Convenient for notebooks/scripts; returns a :class:`CorrectnessReport`.
    """

    return run_single_qubit_correctness_experiment(
        phase_angle=phase_angle,
        shots=shots,
        seed=seed,
    )


def performance_demo(
    *, dims: tuple[int, ...] = (8, 16, 32, 64), stages: int = 16, repeats: int = 3
) -> Dict[str, Any]:
    """Run a simple performance benchmark and return the results.

    The result is a serialisable dictionary with points for the CPU/NumPy
    backend.
    """

    points = run_performance_benchmark(dims=dims, stages=stages, repeats=repeats)
    serializable = [
        {
            "dim": int(p.dim),
            "stages": int(p.stages),
            "backend": p.backend,
            "elapsed_sec": float(p.elapsed_sec),
        }
        for p in points
    ]
    return {"points": serializable}


def batched_performance_demo(
    *,
    dims: tuple[int, ...] = (256, 512, 1024),
    batch_size: int = 128,
    stages: int = 512,
    repeats: int = 3,
) -> Dict[str, Any]:
    """Batched CPU/NumPy performance benchmark.

    Returns a serialisable list of timing points for different dimensions.
    """

    points = run_batched_performance_benchmark(
        dims=dims,
        batch_size=batch_size,
        stages=stages,
        repeats=repeats,
    )
    serializable = [
        {
            "dim": int(p.dim),
            "batch_size": int(p.batch_size),
            "stages": int(p.stages),
            "backend": p.backend,
            "elapsed_sec": float(p.elapsed_sec),
        }
        for p in points
    ]
    return {"points": serializable}


def multi_conjugate_zero_sum_demo(
    *,
    poles: int = 6,
    n_conjugates: int = 4,
    seed: int | None = 7,
) -> Dict[str, Any]:
    """Showcase N‑projection with explicit Σ→0 for k>2.

    Builds a random state with the given number of poles and conjugates,
    applies a zero‑sum projector (subtracting the mean), and returns Σ
    before/after together with state snapshots.
    """

    rng = np.random.default_rng(seed)
    amplitudes = rng.normal(size=poles) + 1j * rng.normal(size=poles)
    mv = MultiConjugateFunction(amplitudes.astype(np.complex128), n_conjugates=n_conjugates)

    # Zero-sum projector: P = I - (1/n) 1 1^T
    eye = np.eye(poles, dtype=np.complex128)
    ones = np.ones((poles, 1), dtype=np.complex128)
    proj = eye - (1.0 / poles) * (ones @ ones.T)

    from .gates import n_project as n_project_func

    res = n_project_func(mv, proj, renormalize=True)

    return {
        "poles": poles,
        "n_conjugates": n_conjugates,
        "sigma_before": complex(res.sigma_before),
        "sigma_after": complex(res.sigma_after),
        "residual": float(res.residual),
        "before_summary": state_summary(mv),
        "after_summary": state_summary(res.state),
    }


__all__ = [
    "hadamard_phase_measure_demo",
    "single_qubit_correctness_demo",
    "performance_demo",
    "batched_performance_demo",
    "multi_conjugate_zero_sum_demo",
]
