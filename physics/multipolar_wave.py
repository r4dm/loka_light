"""Waveform abstractions that track n-conjugate amplitudes, compute probability density, and bridge
between raw arrays and MultipolarValue projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from ..core.value import MultipolarValue
from ..core.polarity import Polarity


@dataclass
class WaveMetadata:
    """Diagnostic bundle describing the loka and Sigma metrics of a wave."""

    loka_name: str
    polarity_names: List[str]
    sigma_norm: float
    sigma_residual: complex


class MultiConjugateFunction:
    """n-conjugate wave preserving Σ balance across all poles."""

    def __init__(
        self,
        amplitudes: Sequence[complex] | MultipolarValue,
        *,
        n_conjugates: int,
        metadata: WaveMetadata | None = None,
    ) -> None:
        if n_conjugates < 2:
            raise ValueError("n_conjugates must be ≥ 2")
        self.n_conjugates = int(n_conjugates)
        self._mv: MultipolarValue | None = None
        self._basis: List[Polarity] | None = None

        if isinstance(amplitudes, MultipolarValue):
            self._mv = amplitudes
            self._basis = list(amplitudes.loka.polarities)
            seq = [amplitudes.coefficients.get(p, 0.0) for p in self._basis]
            self.amplitudes = np.asarray(seq, dtype=np.complex128)
        else:
            arr = np.asarray(list(amplitudes), dtype=np.complex128)
            if arr.ndim != 1:
                raise ValueError("amplitudes must be a 1-D sequence")
            self.amplitudes = arr
        self.metadata = metadata

    # ------------------------------------------------------------------
    def probability_tensor(self) -> np.ndarray:
        """Return the full density matrix ρ = |ψ⟩⟨ψ|.

        The tensor keeps interference information between all poles and
        complements :meth:`probability_density`, which returns only the
        scalar Σ|ψ|² scaled by k/2.
        """

        return np.outer(self.amplitudes, np.conj(self.amplitudes))

    # ------------------------------------------------------------------
    def copy(self) -> "MultiConjugateFunction":
        """Return a duplicate wave, keeping metadata and n-conjugate count."""

        return MultiConjugateFunction(
            self.amplitudes.copy(),
            n_conjugates=self.n_conjugates,
            metadata=self.metadata,
        )

    def normalize(self) -> None:
        """Scale amplitudes so that Σ|ψ|²=1 and refresh the backing MV."""

        norm = np.linalg.norm(self.amplitudes)
        if norm == 0:
            raise ValueError("cannot normalize zero amplitudes")
        self.amplitudes = self.amplitudes / norm
        if self._mv is not None and self._basis is not None:
            coeffs = {p: c for p, c in zip(self._basis, self.amplitudes)}
            self._mv = MultipolarValue(self._mv.loka, coeffs)

    def probability_density(self) -> float:
        """Return Σ|ψ|² scaled by k/2 to match n-conjugate energy rules."""

        base = float(np.sum(np.abs(self.amplitudes) ** 2))
        return base * (self.n_conjugates / 2)

    def collapse(self, *, recursive: bool = True) -> complex:
        """Collapse the wave to a number using the underlying MultipolarValue."""

        if self._mv is None:
            raise ValueError("wave was not built from a MultipolarValue")
        return self._mv.collapse(recursive=recursive)

    def to_multipolar_value(self, loka) -> MultipolarValue:
        """Project amplitudes into the supplied loka's polarity basis."""

        if self._basis is not None and self._mv is not None:
            return MultipolarValue(self._mv.loka, {p: c for p, c in zip(self._basis, self.amplitudes)})
        polarities = getattr(loka, "polarity", None)
        polarities = getattr(loka, "polarities", polarities)
        if polarities is None:
            raise ValueError("loka must expose polarities to convert amplitudes")
        if len(polarities) != len(self.amplitudes):
            raise ValueError("amplitudes length must equal number of polarities")
        coeffs = {polarity: coeff for polarity, coeff in zip(polarities, self.amplitudes)}
        return MultipolarValue(loka, coeffs)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.amplitudes.size

    def __iter__(self) -> Iterable[complex]:  # pragma: no cover
        return iter(self.amplitudes)

    def __repr__(self) -> str:  # pragma: no cover
        return f"MultiConjugateFunction(dim={len(self)}, k={self.n_conjugates})"


__all__ = ["MultiConjugateFunction", "WaveMetadata"]
