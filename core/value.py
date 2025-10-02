"""MultipolarValue container for pole amplitudes, supporting algebraic operations, collapse, and
conjugation while enforcing n-conjugate balance across cascades."""

from __future__ import annotations

import cmath
from typing import Dict, Union

from .loka import Loka
from .polarity import Polarity


class MultipolarValue:
    """Store pole amplitudes for a specific Loka.

    The value tracks coefficients for each polarity, supports algebraic
    operations, and can spawn composite polarities. This mirrors the M/N
    cascade logic used across Sigma-balanced devices.
    """

    def __init__(
        self,
        loka: Loka,
        coefficients: Dict[Union[Polarity, str], Union[float, int, complex]],
    ) -> None:
        if not isinstance(loka, Loka):
            raise TypeError("loka must be an instance of Loka")
        if not isinstance(coefficients, dict):
            raise TypeError("coefficients must be provided as a dict")
        self.loka = loka
        self.coefficients: Dict[Polarity, complex] = {}
        for key, value in coefficients.items():
            if not isinstance(value, (int, float, complex)):
                raise ValueError(f"coefficient for '{key}' must be numeric")
            pol_obj = key if isinstance(key, Polarity) else self.loka.get_polarity_by_name(str(key))
            not_in_loka = pol_obj is None or self.loka.get_polarity_by_name(pol_obj.name) is None
            if not_in_loka:
                if isinstance(key, Polarity) and getattr(key, "source_mv", None) is not None and key.source_mv.loka == self.loka:
                    pol_obj = key
                else:
                    raise ValueError(f"polarity '{key}' is not part of loka '{self.loka.name}'")
            self.coefficients[pol_obj] = self.coefficients.get(pol_obj, complex(0)) + complex(value)

    def __str__(self) -> str:  # pragma: no cover - formatting helper
        if not any(abs(c) > 1e-9 for c in self.coefficients.values()):
            return f"0 (in loka '{self.loka.name}')"
        sorted_terms = []
        for polarity in self.loka.polarities:
            coeff = self.coefficients.get(polarity, 0)
            if abs(coeff) < 1e-9:
                continue
            if isinstance(coeff, complex) and abs(coeff.imag) > 1e-9:
                coeff_str = f"({coeff.real:.4f}{coeff.imag:+.4f}j)*"
            else:
                coeff_real = coeff.real
                if abs(coeff_real - 1.0) < 1e-9:
                    coeff_str = ""
                elif abs(coeff_real + 1.0) < 1e-9:
                    coeff_str = "-"
                else:
                    coeff_str = f"{coeff_real:.4f}*"
            sorted_terms.append(f"{coeff_str}{polarity.name}")
        if not sorted_terms:
            return f"0 (in loka '{self.loka.name}')"
        result_str = " + ".join(sorted_terms).replace("+ -", "- ")
        return result_str + f" (in loka '{self.loka.name}')"

    def __eq__(self, other):
        if not isinstance(other, MultipolarValue) or self.loka.name != other.loka.name:
            return NotImplemented
        all_keys = set(self.coefficients.keys()) | set(other.coefficients.keys())
        for key in all_keys:
            if not cmath.isclose(
                self.coefficients.get(key, 0),
                other.coefficients.get(key, 0),
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                return False
        return True

    def __hash__(self) -> int:
        rounded_items = tuple(
            sorted(
                (polarity.name, round(coeff.real, 12), round(coeff.imag, 12))
                for polarity, coeff in self.coefficients.items()
                if abs(coeff) > 1e-12
            )
        )
        return hash((self.loka.name, rounded_items))

    # -- composite polarity helpers ----------------------------------------
    def to_polarity(self, name: str) -> Polarity:
        """Return a composite polarity backed by this value."""

        max_level = 0
        for polarity in self.coefficients:
            if hasattr(polarity, "level"):
                max_level = max(max_level, polarity.level)
        return Polarity(name, source_mv=self, level=max_level + 1)

    def get_coefficient(self, polarity: Union[Polarity, str]) -> complex:
        """Lookup the complex coefficient for the given polarity."""

        pol_name = polarity.name if isinstance(polarity, Polarity) else polarity
        pol_obj = self.loka.get_polarity_by_name(pol_name)
        if pol_obj is None:
            raise ValueError(f"polarity '{pol_name}' does not belong to loka '{self.loka.name}'")
        return self.coefficients.get(pol_obj, complex(0))

    # -- arithmetic ---------------------------------------------------------
    def __add__(self, other):
        if not isinstance(other, MultipolarValue):
            return NotImplemented
        if self.loka.name != other.loka.name:
            raise TypeError("addition requires both values to belong to the same loka")
        new_coeffs = self.coefficients.copy()
        for polarity, coeff in other.coefficients.items():
            new_coeffs[polarity] = new_coeffs.get(polarity, 0) + coeff
        return MultipolarValue(self.loka, new_coeffs)

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            return MultipolarValue(self.loka, {p: c * other for p, c in self.coefficients.items()})
        if not isinstance(other, MultipolarValue):
            return NotImplemented
        if self.loka.name != other.loka.name:
            raise TypeError("multiplication requires both values to belong to the same loka")
        result_coeffs: Dict[Polarity, complex] = {}
        for p1, c1 in self.coefficients.items():
            for p2, c2 in other.coefficients.items():
                if abs(c1) < 1e-9 or abs(c2) < 1e-9:
                    continue
                product_pol = self.loka.multiply(p1, p2)
                result_coeffs[product_pol] = result_coeffs.get(product_pol, 0) + c1 * c2
        return MultipolarValue(self.loka, result_coeffs)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float, complex)):
            if abs(other) < 1e-9:
                raise ZeroDivisionError("division by zero")
            return self.__mul__(1.0 / other)
        return NotImplemented

    def __neg__(self):
        return self.__mul__(-1)

    def __sub__(self, other):
        return self.__add__(other.__neg__())

    # -- reductions ---------------------------------------------------------
    def collapse(self, recursive: bool = True) -> complex:
        """Collapse the multipolar structure into a single complex number."""

        total_value = complex(0)
        for polarity, coeff in self.coefficients.items():
            value = polarity.value
            if value is None and recursive and getattr(polarity, "source_mv", None) is not None:
                value = polarity.source_mv.collapse(recursive=True)
            if value is None:
                raise ValueError(f"polarity '{polarity.name}' has no numeric value")
            total_value += coeff * value
        return total_value

    def conjugate(self) -> "MultipolarValue":
        """Return the n-conjugate value preserving Sigma balance."""

        new_coeffs: Dict[Polarity, complex] = {}
        for polarity, coeff in self.coefficients.items():
            polarity_internal = self.loka.get_polarity_by_name(polarity.name) or polarity
            conjugated = self.loka.conjugate_polarity(polarity_internal)
            new_coeffs[conjugated] = new_coeffs.get(conjugated, 0) + complex(coeff).conjugate()
        return MultipolarValue(self.loka, new_coeffs)

    def norm(self, mode: str = "numeric"):
        """Compute the norm using Sigma-preserving rules."""

        if mode == "numeric":
            value = self.collapse()
            return float(abs(value) ** 2)
        product = self * self.conjugate()
        if mode == "mv":
            return product
        if mode == "mv_neutral":
            neutral = getattr(self.loka, "neutral_element", None)
            if neutral is None:
                return product
            neutral_coeff = product.coefficients.get(neutral, 0)
            return MultipolarValue(self.loka, {neutral: neutral_coeff})
        raise ValueError("norm mode must be 'numeric', 'mv', or 'mv_neutral'")

    def normalize(self):
        """Wrap coefficients using the loka cyclicity if available."""

        if self.loka.cyclicity is None or self.loka.cyclicity == 0:
            return self
        n = self.loka.cyclicity
        new_coeffs = self.coefficients.copy()
        for polarity, coeff in new_coeffs.items():
            if abs(coeff.imag) < 1e-9:
                new_coeffs[polarity] = complex(coeff.real % n, coeff.imag)
        return MultipolarValue(self.loka, new_coeffs)
