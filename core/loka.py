"""Base Loka abstraction and registry: tracks Tattva/Dharma invariants, validates polarity sets, and
exposes algebraic hooks that preserve Σ→0 while enumerating admissible intensity modes for minds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, TYPE_CHECKING

from .polarity import Polarity

if TYPE_CHECKING:  # pragma: no cover
    from .value import MultipolarValue


@dataclass(frozen=True)
class DharmaProfile:
    """Describes a law that controls how a Loka balances Sigma->0."""

    name: str
    relations: Tuple[str, ...]
    polarity_arity: int
    theory_refs: Tuple[str, ...]
    description: str = ""
    intensity_mode: str = "sigma_add"


@dataclass(frozen=True)
class TattvaProfile:
    """Defines the base pole space used by a mind and its cascades."""

    name: str
    polarity_count: int
    dharmas: Tuple[str, ...]
    theory_refs: Tuple[str, ...]
    description: str = ""
    mind_modes: Tuple[str, ...] = ()
    intensity_modes: Tuple[str, ...] = ("sigma_add",)


REGISTERED_DHARMAS: Tuple[DharmaProfile, ...] = (
    DharmaProfile(
        name="unit_conservation_kernel",
        relations=("Sigma->0",),
        polarity_arity=1,
        theory_refs=("sigma_conservation",),
        description="Single-pole conservation keeps cascades at Sigma->0.",
        intensity_mode="sigma_conserve",
    ),
    DharmaProfile(
        name="binary_linear_kernel",
        relations=("(-)(-) -> +", "(-)(+) -> -", "(+)(-) -> -", "(+)(+) -> +"),
        polarity_arity=2,
        theory_refs=("binary_balance",),
        description="Binary linear mind preserves additive balance for plus/minus poles.",
        intensity_mode="sigma_add",
    ),
    DharmaProfile(
        name="triadic_superposition_kernel",
        relations=("i*j*k -> Om", "i^2 -> Om", "j^2 -> Om", "k^2 -> Om"),
        polarity_arity=3,
        theory_refs=("triadic_superposition",),
        description="Triadic superposition couples three poles into a shared neutral state.",
        intensity_mode="sigma_mul",
    ),
    DharmaProfile(
        name="complex_wave_kernel",
        relations=("exp(i*phi)", "r^2 = ln(1 + w/eps)"),
        polarity_arity=4,
        theory_refs=("complex_wave_behaviour",),
        description="Four-pole complex waves mix amplitude and phase in the Sigma ladder.",
        intensity_mode="sigma_phase",
    ),
    DharmaProfile(
        name="harloka_bundle_kernel",
        relations=("har(p) <-> C3", "har(q) <-> C4"),
        polarity_arity=4,
        theory_refs=("harloka_bundle",),
        description="Harloka bundles align triadic and tetradic spaces inside one cascade.",
        intensity_mode="sigma_mul",
    ),
    DharmaProfile(
        name="relational_visibility_kernel",
        relations=("A & B -> O", "A & !B -> A", "!A & B -> B"),
        polarity_arity=2,
        theory_refs=("visibility_rules",),
        description="Visibility law toggles output poles using relational gates.",
        intensity_mode="sigma_gate",
    ),
    DharmaProfile(
        name="hyper_wave_kernel",
        relations=("Sigma_k > 0", "parity_switch"),
        polarity_arity=11,
        theory_refs=("hyper_wave_patterns",),
        description="Hyper waves alternate superposition and visibility across high-rank poles.",
        intensity_mode="sigma_phase",
    ),
)

DHARMA_REGISTRY: Dict[str, DharmaProfile] = {}
TATTVA_REGISTRY: Dict[str, TattvaProfile] = {}


def register_dharma(profile: DharmaProfile) -> None:
    """Register a Dharma profile, validating uniqueness."""

    existing = DHARMA_REGISTRY.get(profile.name)
    if existing and existing != profile:
        raise ValueError(f"Dharma '{profile.name}' is already registered with different contents")
    DHARMA_REGISTRY[profile.name] = profile


def get_dharma(name: str) -> DharmaProfile:
    """Fetch a Dharma profile by name."""

    if name not in DHARMA_REGISTRY:
        raise KeyError(f"Dharma '{name}' is not registered")
    return DHARMA_REGISTRY[name]


def list_dharmas() -> Tuple[DharmaProfile, ...]:
    """Return all registered Dharma profiles sorted by name."""

    return tuple(DHARMA_REGISTRY[key] for key in sorted(DHARMA_REGISTRY))


def register_tattva(profile: TattvaProfile) -> None:
    """Register a Tattva and ensure its Dharmas already exist."""

    for dharma_name in profile.dharmas:
        if dharma_name not in DHARMA_REGISTRY:
            raise KeyError(
                f"Dharma '{dharma_name}' must be registered before tattva '{profile.name}'"
            )
    existing = TATTVA_REGISTRY.get(profile.name)
    if existing and existing != profile:
        raise ValueError(f"Tattva '{profile.name}' is already registered with different contents")
    TATTVA_REGISTRY[profile.name] = profile


def get_tattva(name: str) -> TattvaProfile:
    """Fetch a Tattva profile by name."""

    if name not in TATTVA_REGISTRY:
        raise KeyError(f"Tattva '{name}' is not registered")
    return TATTVA_REGISTRY[name]


def list_tattvas() -> Tuple[TattvaProfile, ...]:
    """Return all registered Tattva profiles sorted by name."""

    return tuple(TATTVA_REGISTRY[key] for key in sorted(TATTVA_REGISTRY))


def _normalize_refs(refs: Iterable[str]) -> Tuple[str, ...]:
    ordered: Dict[str, None] = {}
    for ref in refs:
        ordered.setdefault(ref, None)
    return tuple(ordered.keys())


def _seed_default_registries() -> None:
    if DHARMA_REGISTRY:
        return
    for profile in REGISTERED_DHARMAS:
        register_dharma(profile)
    tattvas = (
        TattvaProfile(
            name="unit_conserve",
            polarity_count=1,
            dharmas=("unit_conservation_kernel",),
            theory_refs=("041", "045"),
            description="Single-pole lattices that enforce Sigma->0.",
            mind_modes=("singleton",),
        ),
        TattvaProfile(
            name="binary_linear",
            polarity_count=2,
            dharmas=("unit_conservation_kernel", "binary_linear_kernel"),
            theory_refs=("binary_balance",),
            description="Two-pole mind with additive law and Sigma guard.",
            mind_modes=("binary",),
        ),
        TattvaProfile(
            name="triadic_superposition",
            polarity_count=3,
            dharmas=("triadic_superposition_kernel",),
            theory_refs=("triadic_superposition",),
            description="Three-pole superposition used for triadic cascades.",
            mind_modes=("triadic",),
            intensity_modes=("sigma_mul", "sigma_add"),
        ),
        TattvaProfile(
            name="complex_wave",
            polarity_count=4,
            dharmas=("complex_wave_kernel",),
            theory_refs=("complex_wave_behaviour",),
            description="Four-pole complex wave base with amplitude-phase balance.",
            mind_modes=("complex", "wave"),
            intensity_modes=("sigma_phase", "sigma_add", "sigma_mul"),
        ),
        TattvaProfile(
            name="quaternion_harloka",
            polarity_count=4,
            dharmas=("triadic_superposition_kernel", "complex_wave_kernel", "harloka_bundle_kernel"),
            theory_refs=("harloka_bundle",),
            description="Harloka bundle linking triadic and tetradic poles.",
            mind_modes=("harloka",),
            intensity_modes=("sigma_mul", "sigma_phase"),
        ),
        TattvaProfile(
            name="relational_visibility",
            polarity_count=4,
            dharmas=("relational_visibility_kernel",),
            theory_refs=("visibility_rules",),
            description="Visibility gating across four-node cascades.",
            mind_modes=("visibility",),
            intensity_modes=("sigma_gate", "sigma_add"),
        ),
        TattvaProfile(
            name="hyper_wave",
            polarity_count=11,
            dharmas=("hyper_wave_kernel",),
            theory_refs=("hyper_wave_patterns",),
            description="High-rank hyper wave cascades with alternating parity.",
            mind_modes=("hyper",),
            intensity_modes=("sigma_phase", "sigma_add", "sigma_mul", "sigma_gate"),
        ),
    )
    for tattva in tattvas:
        register_tattva(tattva)


_seed_default_registries()


def _resolve_tattva_spec(spec: TattvaProfile | str | None) -> TattvaProfile | None:
    if spec is None:
        return None
    if isinstance(spec, TattvaProfile):
        return spec
    return get_tattva(str(spec))


def _resolve_dharma_specs(
    specs: Sequence[DharmaProfile | str] | None,
    tattva: TattvaProfile | None,
) -> Tuple[DharmaProfile, ...]:
    if specs is None:
        if tattva is None:
            return ()
        specs = tattva.dharmas

    resolved: List[DharmaProfile] = []
    for item in specs:
        if isinstance(item, DharmaProfile):
            resolved.append(item)
        else:
            resolved.append(get_dharma(str(item)))
    return tuple(resolved)


class Loka:
    """Base algebra for multipolar worlds.

    A Loka keeps explicit poles, registered Dharmas, and Tattva metadata so
    cascades can verify Sigma->0 balance and n-conjugacy constraints. Specific
    algebras implement multiply, add, divide, and inverse for their pole set.
    """

    def __init__(
        self,
        name: str,
        polarities: List[Polarity],
        *,
        tattva: TattvaProfile | str | None = None,
        dharmas: Sequence[DharmaProfile | str] | None = None,
        mind_modes: Sequence[str] | None = None,
        intensity_modes: Sequence[str] | None = None,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("loka name must be a non-empty string")
        if not polarities or not all(isinstance(p, Polarity) for p in polarities):
            raise ValueError("loka requires a list of Polarity instances")
        if len(polarities) != len({p.name for p in polarities}):
            raise ValueError(f"polarity names inside '{name}' must be unique")

        self.name = name
        self.polarities = tuple(polarities)
        self.rank = len(polarities)
        self._polarity_map = {p.name: p for p in self.polarities}
        self.cyclicity: int | None = None
        self.neutral_element: Polarity | None = None

        self.tattva_profile = _resolve_tattva_spec(tattva)
        self.dharmas: Tuple[DharmaProfile, ...] = _resolve_dharma_specs(dharmas, self.tattva_profile)

        mind_pool: List[str] = []
        if self.tattva_profile:
            mind_pool.extend(self.tattva_profile.mind_modes)
        if mind_modes is not None:
            mind_pool.extend(str(mode) for mode in mind_modes)
        self.mind_modes = tuple(dict.fromkeys(mind_pool))

        intensity_pool: List[str] = []
        if self.tattva_profile:
            intensity_pool.extend(self.tattva_profile.intensity_modes)
        if intensity_modes is not None:
            intensity_pool.extend(str(mode) for mode in intensity_modes)
        for dharma in self.dharmas:
            intensity_pool.append(dharma.intensity_mode)
        self.intensity_modes = tuple(dict.fromkeys(intensity_pool))

        theory_refs: List[str] = []
        if self.tattva_profile:
            theory_refs.extend(self.tattva_profile.theory_refs)
        for dharma in self.dharmas:
            theory_refs.extend(dharma.theory_refs)
        self.theory_refs = _normalize_refs(theory_refs)

    def get_polarity_by_name(self, name: str) -> Polarity | None:
        """Return the polarity registered for the given name."""

        return self._polarity_map.get(name)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"{self.__class__.__name__}(name={self.name!r}, rank={self.rank})"

    def supports_intensity(self, mode: str) -> bool:
        """Check whether the loka allows the requested intensity mode."""

        return mode in self.intensity_modes

    def structural_passport(self) -> Dict[str, Any]:
        """Return a summary compatible with existing structural passports."""

        return {
            "name": self.name,
            "rank": self.rank,
            "tattva": self.tattva_profile.name if self.tattva_profile else None,
            "dharmas": [d.name for d in self.dharmas],
            "mind_modes": list(self.mind_modes),
            "intensity_modes": list(self.intensity_modes),
            "theory_refs": list(self.theory_refs),
        }

    def _check_operands(self, p1: Polarity, p2: Polarity) -> Tuple[Polarity, Polarity]:
        if not isinstance(p1, Polarity) or not isinstance(p2, Polarity):
            raise TypeError("operands must be Polarity instances")
        pol1_internal = self._polarity_map.get(p1.name)
        pol2_internal = self._polarity_map.get(p2.name)
        if not pol1_internal:
            raise ValueError(f"operand '{p1.name}' not found in loka '{self.name}'")
        if not pol2_internal:
            raise ValueError(f"operand '{p2.name}' not found in loka '{self.name}'")
        return pol1_internal, pol2_internal

    # -- core algebra hooks -------------------------------------------------
    def multiply(self, p1: Polarity, p2: Polarity) -> Polarity:
        raise NotImplementedError

    def add(self, p1: Polarity, p2: Polarity) -> Polarity:
        raise NotImplementedError

    def inverse(self, polarity: Polarity) -> Polarity:
        raise NotImplementedError

    def divide(self, numerator: Polarity, denominator: Polarity) -> Polarity:
        raise NotImplementedError

    # -- lookup utilities ---------------------------------------------------
    def cayley_table(self, operation: str = "multiply") -> List[List[str]]:
        if operation not in ("multiply", "add"):
            raise ValueError("operation must be 'multiply' or 'add'")
        table: List[List[str]] = []
        for p1 in self.polarities:
            row: List[str] = []
            for p2 in self.polarities:
                op = getattr(self, operation)
                result = op(p1, p2)
                row.append(result.name)
            table.append(row)
        return table

    def evaluate_sequence(self, polarities_sequence: List[Polarity], operation_type: str) -> Polarity:
        if not polarities_sequence:
            raise ValueError("polarity sequence cannot be empty")
        op_map = {"multiply": self.multiply, "add": self.add, "divide": self.divide}
        if operation_type not in op_map:
            raise ValueError(f"unknown operation type: {operation_type}")
        current, _ = self._check_operands(polarities_sequence[0], polarities_sequence[0])
        for polarity in polarities_sequence[1:]:
            current = op_map[operation_type](current, polarity)
        return current

    def conjugate_polarity(self, polarity: Polarity) -> Polarity:
        try:
            return self.inverse(polarity)
        except Exception:
            return self._polarity_map.get(polarity.name, polarity)

    def to_metadata(self) -> Dict[str, Any]:
        """Serialize basic Loka information for reports and runs."""

        data = self.structural_passport()
        data.update(
            {
                "polarities": [p.name for p in self.polarities],
                "cyclicity": self.cyclicity,
                "neutral_element": self.neutral_element.name if self.neutral_element else None,
            }
        )
        return data

    # -- numeric helpers ----------------------------------------------------
    def scalar_weight(self, polarity: Polarity) -> complex:
        pol = self._polarity_map.get(polarity.name)
        if pol is None:
            raise ValueError(f"polarity '{polarity.name}' is not part of '{self.name}'")
        if pol.value is None:
            raise ValueError(f"polarity '{polarity.name}' has no numeric weight")
        return complex(pol.value)

    def average_value(self) -> complex:
        numeric: List[complex] = []
        for p in self.polarities:
            if p.value is not None:
                numeric.append(complex(p.value))
        if not numeric:
            raise ValueError("no numeric polarities to average")
        return sum(numeric) / len(numeric)

    def basis_sum(self) -> complex:
        """Return the sum of *basis numeric weights* for this loka.

        This is the sum of the numeric projections stored on :class:`Polarity`
        objects (e.g. roots of unity for ``LokaCn``). It is **not** the Σ used
        in M/N cascades and `physics.sigma`, where Σ is defined as the sum of
        *amplitudes/coefficients* of a concrete wave/value.
        """

        total = complex(0)
        for p in self.polarities:
            if p.value is not None:
                total += complex(p.value)
        return total

    def sigma_balance(self) -> complex:
        """Alias for :meth:`basis_sum` (kept for backward compatibility)."""

        return self.basis_sum()

    # -- validation ---------------------------------------------------------
    def ensure_sigma_zero(self, tolerance: float = 1e-9) -> None:
        """Validate that :meth:`basis_sum` is ~0 within tolerance.

        Note: for amplitude Σ checks (Σ→0 purification), use `physics.sigma`.
        """

        basis = self.basis_sum()
        if abs(basis) > tolerance:
            raise ValueError(f"Basis sum {basis} exceeds tolerance {tolerance}")

    # -- convenience wrappers -----------------------------------------------
    def is_invertible(self) -> bool:
        try:
            for p in self.polarities:
                self.inverse(p)
        except NotImplementedError:
            return False
        except Exception:
            return False
        return True

    def supports_mind(self, mind_name: str) -> bool:
        return mind_name in self.mind_modes

    def describe(self) -> str:
        lines = [f"Loka {self.name} (rank={self.rank})"]
        lines.append(f"polarities: {', '.join(p.name for p in self.polarities)}")
        if self.tattva_profile:
            lines.append(f"tattva: {self.tattva_profile.name}")
        if self.dharmas:
            lines.append("dharmas: " + ", ".join(d.name for d in self.dharmas))
        if self.intensity_modes:
            lines.append("intensity modes: " + ", ".join(self.intensity_modes))
        return "\n".join(lines)
