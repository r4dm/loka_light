"""Quantum utilities for multipolar spaces: entanglement poles, bounded spectra, Hilbert operators, and
Schrodinger solvers that maintain n-conjugate norms."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np

from ..core.loka import Loka
from ..core.polarity import Polarity
from ..core.value import MultipolarValue


# ---------------------------------------------------------------------------
# Entanglement primitives
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _cached_basis(n: int) -> Loka:
    polarities = [Polarity(str(i)) for i in range(n)]
    return Loka(name=f"QBasis{n}", polarities=polarities)


@dataclass
class EntanglementPole:
    """Normalised entangled state with Σ|a_i|² = 1."""

    amplitudes: np.ndarray

    def __post_init__(self) -> None:
        arr = np.asarray(self.amplitudes, dtype=np.complex128)
        if arr.ndim != 1:
            raise ValueError("amplitudes must be a 1-D array")
        norm = np.linalg.norm(arr)
        if norm == 0:
            raise ValueError("state vector cannot be zero")
        self.amplitudes = arr / norm

    @property
    def n(self) -> int:
        return self.amplitudes.size

    def as_multipolar_value(self) -> MultipolarValue:
        loka = _cached_basis(self.n)
        coeffs = {loka.polarities[i]: self.amplitudes[i] for i in range(self.n)}
        return MultipolarValue(loka, coeffs)

    def tensor(self, other: "EntanglementPole") -> "EntanglementPole":
        return EntanglementPole(np.kron(self.amplitudes, other.amplitudes))

    def measure(self, rng: np.random.Generator | None = None, *, collapse: bool = False) -> int:
        rng = rng or np.random.default_rng()
        probs = np.abs(self.amplitudes) ** 2
        outcome = int(rng.choice(self.n, p=probs))
        if collapse:
            collapsed = np.zeros_like(self.amplitudes)
            collapsed[outcome] = 1.0
            self.amplitudes = collapsed
        return outcome

    def apply_unitary(self, operator: np.ndarray) -> "EntanglementPole":
        op = np.asarray(operator, dtype=np.complex128)
        if op.shape != (self.n, self.n):
            raise ValueError("unitary matrix dimension mismatch")
        if not np.allclose(op.conj().T @ op, np.eye(self.n), atol=1e-8):
            raise ValueError("matrix is not unitary")
        return EntanglementPole(op @ self.amplitudes)

    def __array__(self, dtype=None):  # pragma: no cover
        return self.amplitudes.astype(dtype) if dtype else self.amplitudes

    def __repr__(self) -> str:  # pragma: no cover
        return f"EntanglementPole({np.array2string(self.amplitudes, precision=3)})"


def finite_spectrum(operator: np.ndarray, max_eigenvalue: float = 1.0) -> np.ndarray:
    op = np.asarray(operator, dtype=np.complex128)
    if op.shape[0] != op.shape[1]:
        raise ValueError("operator must be square")
    if not np.allclose(op, op.conj().T, atol=1e-8):
        raise ValueError("operator must be Hermitian")
    eigenvalues = np.linalg.eigvalsh(op)
    bound = np.max(np.abs(eigenvalues))
    if bound <= max_eigenvalue:
        return op
    return (max_eigenvalue / bound) * op


# ---------------------------------------------------------------------------
# Hilbert space helpers
# ---------------------------------------------------------------------------


class PoleHilbertSpace:
    """Finite-dimensional Hilbert space aligned with multipolar loci."""

    def __init__(self, dimension: int, n_poles: int, *, dtype: np.dtype = np.complex128) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if n_poles <= 0:
            raise ValueError("n_poles must be positive")
        self.dimension = int(dimension)
        self.n_poles = int(n_poles)
        self.dtype = dtype
        self._basis = np.eye(self.dimension, dtype=self.dtype)

    def basis_vector(self, index: int) -> np.ndarray:
        if not 0 <= index < self.dimension:
            raise IndexError("basis index out of range")
        return self._basis[index].copy()

    def normalize(self, state: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(state)
        if norm == 0:
            raise ValueError("cannot normalize zero state")
        return state / norm

    def random_state(self, *, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dimension) + 1j * rng.standard_normal(self.dimension)
        return self.normalize(vec.astype(self.dtype))

    def to_array(self, state: Union[np.ndarray, MultipolarValue]) -> np.ndarray:
        if isinstance(state, np.ndarray):
            if state.shape != (self.dimension,):
                raise ValueError("state dimension mismatch")
            return state.astype(self.dtype, copy=False)
        if len(state.loka.polarities) != self.dimension:
            raise ValueError("loka polarity count must equal space dimension")
        coeffs = [state.coefficients.get(polarity, 0.0) for polarity in state.loka.polarities]
        return np.asarray(coeffs, dtype=self.dtype)

    def from_array(self, arr: np.ndarray, template: Union[np.ndarray, MultipolarValue]) -> Union[np.ndarray, MultipolarValue]:
        if arr.shape != (self.dimension,):
            raise ValueError("array length mismatch")
        if isinstance(template, np.ndarray):
            return arr.astype(self.dtype, copy=False)
        coeffs = {
            polarity: value
            for polarity, value in zip(template.loka.polarities, arr)
            if abs(value) > 1e-12
        }
        return MultipolarValue(template.loka, coeffs)


class PoleOperator:
    """Linear operator acting on a :class:`PoleHilbertSpace`."""

    def __init__(self, space: PoleHilbertSpace, matrix: np.ndarray) -> None:
        mat = np.asarray(matrix, dtype=space.dtype)
        if mat.shape != (space.dimension, space.dimension):
            raise ValueError("matrix dimension mismatch")
        self.space = space
        self.matrix = mat

    def evolve(self, state: np.ndarray, dt: float) -> np.ndarray:
        return (np.eye(self.space.dimension, dtype=self.space.dtype) - 1j * dt * self.matrix) @ state


@dataclass
class GeneralisedSchrodingerEvolution:
    """Discrete evolution that preserves n-conjugate norm."""

    space: PoleHilbertSpace
    operator: PoleOperator
    n_conjugate: int = 2

    def step(self, state: np.ndarray, dt: float) -> np.ndarray:
        evolved = self.operator.evolve(state, dt)
        amps = [complex(a) for a in evolved]
        norm = _normalise_n(amps, self.n_conjugate)
        return np.asarray(norm, dtype=self.space.dtype)


def _normalise_n(values: Sequence[complex], n_conjugate: int) -> List[complex]:
    if n_conjugate == 2:
        vec = np.asarray(values, dtype=np.complex128)
        return list(vec / np.linalg.norm(vec))
    magnitude = sum(abs(v) ** n_conjugate for v in values) ** (1.0 / n_conjugate)
    if magnitude == 0:
        raise ValueError("cannot normalise zero vector")
    return [v / magnitude for v in values]


class PoleSchrodingerSolver:
    """Exact evolution via diagonalisation with optional multipolar scaling."""

    def __init__(
        self,
        operator: PoleOperator,
        *,
        n_conjugates: int = 2,
        modulus_mode: str = "standard",
    ) -> None:
        if n_conjugates < 2:
            raise ValueError("n_conjugates must be ≥ 2")
        self.operator = operator
        self.n_conjugates = int(n_conjugates)
        self.modulus_mode = modulus_mode.lower()
        self._eigvals, self._eigvecs = np.linalg.eigh(operator.matrix)

    def _project(self, state: np.ndarray) -> np.ndarray:
        return self._eigvecs.conj().T @ state

    def _reconstruct(self, coeffs: np.ndarray) -> np.ndarray:
        return self._eigvecs @ coeffs

    def evolve(self, state: Union[np.ndarray, MultipolarValue], t: float, *, scale: float | None = None):
        space = self.operator.space
        state_vec = space.to_array(state)
        coeffs = self._project(state_vec)
        phase = np.exp(-1j * self._eigvals * (scale if scale is not None else 1.0) * t)
        evolved_vec = self._reconstruct(phase * coeffs)
        if isinstance(state, np.ndarray):
            return evolved_vec
        return space.from_array(evolved_vec, state)

    def simulate(self, state0: Union[np.ndarray, MultipolarValue], timeline: Iterable[float]):
        space = self.operator.space
        state_vec = space.to_array(state0)
        coeffs = self._project(state_vec)
        modulus = np.sum(np.abs(state_vec) ** 2)
        if self.n_conjugates > 2 and self.modulus_mode == "generalised":
            modulus = sum(abs(v) ** self.n_conjugates for v in state_vec) ** (2.0 / self.n_conjugates)
        scale = modulus ** (self.n_conjugates / 2 - 1) if self.n_conjugates > 2 else 1.0
        outputs: List[Union[np.ndarray, MultipolarValue]] = []
        for t in timeline:
            phase = np.exp(-1j * self._eigvals * scale * t)
            evolved_vec = self._reconstruct(phase * coeffs)
            outputs.append(space.from_array(evolved_vec, state0))
        return outputs


__all__ = [
    "EntanglementPole",
    "finite_spectrum",
    "PoleHilbertSpace",
    "PoleOperator",
    "GeneralisedSchrodingerEvolution",
    "PoleSchrodingerSolver",
]
