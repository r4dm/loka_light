"""Correctness and performance metrics for the pseudo‑quantum simulator.

Includes:
- variation distance between measurement distributions;
- wrappers for comparison with a simple complex‑valued reference model;
- rough CPU/NumPy benchmarks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from ...physics.multipolar_wave import MultiConjugateFunction

from .circuit import Circuit, GateSpec, run_circuit_cpu
from .gates import measure_polarity


def variation_distance(p: Sequence[float], q: Sequence[float]) -> float:
    """Total variation distance between two discrete distributions.

    ``d_V(p, q) = 0.5 * Σ_i |p_i - q_i|``.
    """

    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    if p_arr.shape != q_arr.shape:
        raise ValueError("p and q must have the same shape")
    return float(0.5 * np.sum(np.abs(p_arr - q_arr)))


def _single_qubit_reference_probs(phase_angle: float) -> np.ndarray:
    """Reference distribution for H → phase → H → measure.

    Computed in a standard 2‑dimensional complex model.
    """

    H = (1.0 / np.sqrt(2.0)) * np.array(
        [[1.0, 1.0],
         [1.0, -1.0]],
        dtype=np.complex128,
    )
    psi0 = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    psi1 = H @ psi0
    phase = np.array([0.0, phase_angle], dtype=float)
    psi2 = psi1 * np.exp(1j * phase)
    psi3 = H @ psi2
    probs = np.abs(psi3) ** 2
    return probs / probs.sum()


@dataclass
class CorrectnessReport:
    phase_angle: float
    shots: int
    empirical_probs: Tuple[float, float]
    reference_probs: Tuple[float, float]
    variation_distance: float


def run_single_qubit_correctness_experiment(
    *,
    phase_angle: float,
    shots: int,
    seed: int | None = None,
) -> CorrectnessReport:
    """Compare H→phase→H→measure outcomes with the analytic reference.

    Uses the same scenario as :func:`hadamard_phase_measure_demo`, but returns
    a :class:`CorrectnessReport` with the variation distance instead of a
    histogram.
    """

    rng = np.random.default_rng(seed)

    from .examples import hadamard_phase_measure_demo

    summary = hadamard_phase_measure_demo(
        shots=shots,
        phase_angle=phase_angle,
        seed=seed,
    )
    counts = summary["counts"]
    total = float(sum(counts.values()))
    emp = np.array([
        counts.get(0, 0) / total,
        counts.get(1, 0) / total,
    ])

    ref = _single_qubit_reference_probs(phase_angle)
    dist = variation_distance(emp, ref)

    return CorrectnessReport(
        phase_angle=float(phase_angle),
        shots=shots,
        empirical_probs=(float(emp[0]), float(emp[1])),
        reference_probs=(float(ref[0]), float(ref[1])),
        variation_distance=dist,
    )


@dataclass
class PerformancePoint:
    dim: int
    stages: int
    backend: str
    elapsed_sec: float


def run_performance_benchmark(
    *,
    dims: Iterable[int] = (8, 16, 32, 64),
    stages: int = 16,
    repeats: int = 3,
) -> List[PerformancePoint]:
    """Rough CPU/NumPy benchmark.

    For each ``dim`` builds a random unitary‑like matrix and runs a chain of
    ``stages`` linear gates on CPU.
    """

    results: List[PerformancePoint] = []

    # CPU / NumPy через MultiConjugateFunction
    from .gates import apply_unitary

    for dim in dims:
        dim = int(dim)
        matrix = _random_unitary_like(dim)
        psi0 = MultiConjugateFunction(
            amplitudes=_random_state(dim),
            n_conjugates=2,
        )

        best = None
        for _ in range(repeats):
            state = psi0
            start = time.perf_counter()
            for _stage in range(stages):
                state = apply_unitary(state, matrix)
            elapsed = time.perf_counter() - start
            best = elapsed if best is None or elapsed < best else best
        assert best is not None
        results.append(PerformancePoint(dim=dim, stages=stages, backend="numpy", elapsed_sec=best))

    return results


def _random_state(dim: int) -> np.ndarray:
    vec = np.random.default_rng().normal(size=dim) + 1j * np.random.default_rng().normal(size=dim)
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    return (vec / norm).astype(np.complex128)


def _random_unitary_like(dim: int) -> np.ndarray:
    """Construct a simple unitary‑like matrix via QR of a random complex matrix."""

    rng = np.random.default_rng()
    real = rng.normal(size=(dim, dim))
    imag = rng.normal(size=(dim, dim))
    mat = real + 1j * imag
    q, r = np.linalg.qr(mat)
    # нормируем диагональ, чтобы получить унитароподобную матрицу
    d = np.diag(r)
    phases = d / np.abs(d)
    return (q * phases).astype(np.complex128)


__all__ = [
    "variation_distance",
    "CorrectnessReport",
    "run_single_qubit_correctness_experiment",
    "PerformancePoint",
    "run_performance_benchmark",
]


@dataclass
class BatchPerformancePoint:
    dim: int
    batch_size: int
    stages: int
    backend: str
    elapsed_sec: float


def run_batched_performance_benchmark(
    *,
    dims: Iterable[int] = (256, 512, 1024),
    batch_size: int = 128,
    stages: int = 512,
    repeats: int = 3,
) -> List[BatchPerformancePoint]:
    """Batched matrix×matrix CPU/NumPy benchmark.

    Uses a batch of states of size ``dim``; same backend as
    :func:`run_performance_benchmark` but more friendly to linear algebra
    libraries.
    """

    results: List[BatchPerformancePoint] = []

    # --- NumPy / CPU -------------------------------------------------------
    for dim in dims:
        dim = int(dim)
        matrix = _random_unitary_like(dim)
        rng = np.random.default_rng()
        states = (
            rng.normal(size=(dim, batch_size))
            + 1j * rng.normal(size=(dim, batch_size))
        ).astype(np.complex128)

        best = None
        for _ in range(repeats):
            x = states
            start = time.perf_counter()
            for _stage in range(stages):
                x = matrix @ x  # (dim, dim) @ (dim, batch)
            elapsed = time.perf_counter() - start
            best = elapsed if best is None or elapsed < best else best
        assert best is not None
        results.append(
            BatchPerformancePoint(
                dim=dim,
                batch_size=batch_size,
                stages=stages,
                backend="numpy",
                elapsed_sec=best,
            )
        )

    return results


__all__ += [
    "BatchPerformancePoint",
    "run_batched_performance_benchmark",
]
