"""Waveform abstractions that track n-conjugate amplitudes, compute probability density, and bridge
between raw arrays and MultipolarValue projections."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, List, Sequence

import numpy as np

from ..core.value import MultipolarValue
from ..core.polarity import Polarity


@dataclass
class WaveMetadata:
    """Diagnostic bundle describing the loka and Σ-metrics of a wave.

    Conventions
    - Σ is defined as the sum of complex amplitudes over all poles.
    - ``sigma_residual`` stores Σ itself (complex).
    - ``sigma_norm`` stores |Σ| (magnitude), useful as a residual for Σ→0 checks.
    """

    loka_name: str
    polarity_names: List[str]
    sigma_norm: float
    sigma_residual: complex
    frequency_hz: float | None = None

    @classmethod
    def from_amplitudes(
        cls,
        amplitudes: Sequence[complex] | np.ndarray,
        *,
        loka_name: str,
        polarity_names: Sequence[str],
        frequency_hz: float | None = None,
    ) -> "WaveMetadata":
        vec = np.asarray(amplitudes, dtype=np.complex128)
        sigma = complex(vec.sum())
        freq = None if frequency_hz is None else float(frequency_hz)
        return cls(
            loka_name=str(loka_name),
            polarity_names=[str(name) for name in polarity_names],
            sigma_norm=float(abs(sigma)),
            sigma_residual=sigma,
            frequency_hz=freq,
        )


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
            if metadata is None:
                metadata = WaveMetadata.from_amplitudes(
                    self.amplitudes,
                    loka_name=amplitudes.loka.name,
                    polarity_names=[p.name for p in amplitudes.loka.polarities],
                )
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
        complements :meth:`probability_density`, which returns the scalar
        generalised modulus power Σ|ψ|^k (k = n_conjugates).
        """

        return np.outer(self.amplitudes, np.conj(self.amplitudes))

    # ------------------------------------------------------------------
    def copy(self) -> "MultiConjugateFunction":
        """Return a duplicate wave, keeping metadata and n-conjugate count."""

        meta = None if self.metadata is None else replace(self.metadata)
        return MultiConjugateFunction(
            self.amplitudes.copy(),
            n_conjugates=self.n_conjugates,
            metadata=meta,
        )

    def normalize(self) -> None:
        """Scale amplitudes so that Σ|ψ|^k = 1 (k = n_conjugates).

        For k=2 this reduces to standard L2 normalisation. For k>2 it matches
        the multi-conjugate modulus rule used across the theory docs.
        """

        k = int(self.n_conjugates)
        norm = self.probability_density()
        if norm == 0.0:
            raise ValueError("cannot normalize zero amplitudes")
        scale = norm ** (1.0 / k)
        self.amplitudes = self.amplitudes / scale
        if self._mv is not None and self._basis is not None:
            coeffs = {p: c for p, c in zip(self._basis, self.amplitudes)}
            self._mv = MultipolarValue(self._mv.loka, coeffs)
        if self.metadata is not None:
            self.metadata = WaveMetadata.from_amplitudes(
                self.amplitudes,
                loka_name=self.metadata.loka_name,
                polarity_names=self.metadata.polarity_names,
                frequency_hz=self.metadata.frequency_hz,
            )

    def probability_density(self) -> float:
        """Return the generalised modulus power Σ|ψ|^k (k = n_conjugates)."""

        k = int(self.n_conjugates)
        return float(np.sum(np.abs(self.amplitudes) ** k))

    def conjugacy_density(self, power: int | None = None) -> float:
        """Return the k-conjugacy density Σ|ψ|^k (alias for :meth:`probability_density`).

        When ``power`` is provided, compute Σ|ψ|^power instead of the default
        ``n_conjugates``.
        """

        p = self.n_conjugates if power is None else int(power)
        if p <= 0:
            raise ValueError("power must be a positive integer")
        return float(np.sum(np.abs(self.amplitudes) ** p))

    def conjugacy_norm(self, power: int | None = None) -> float:
        """Return the Lp-norm (Σ|ψ|^p)^(1/p) for the conjugacy metric."""

        p = self.n_conjugates if power is None else int(power)
        if p <= 0:
            raise ValueError("power must be a positive integer")
        return self.conjugacy_density(power=p) ** (1.0 / float(p))

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
