"""Canonical loka families and superposition helpers: cyclic C_n spaces, harloka bundles, and relation
specs that weave multiple pole spaces into cascade-ready algebras."""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple, Union

from .loka import DharmaProfile, Loka, TattvaProfile
from .polarity import Polarity
from .value import MultipolarValue


@dataclass(frozen=True)
class SuperpositionLayer:
    """Describe how a source Loka maps its poles into the resulting space."""

    name: str
    source_loka: Loka
    mapping: Mapping[str, str]
    intensity_mode: str = "sigma_add"
    description: str = ""

    def normalized_mapping(self) -> Dict[str, str]:
        """Return the mapping with canonical string keys and values."""

        return {str(src): str(dst) for src, dst in self.mapping.items()}


@dataclass(frozen=True)
class HarBundle:
    """Represent a fiber bundle used by harloka constructions."""

    name: str
    fiber_loka: Loka
    projection: Mapping[str, str]
    description: str = ""


@dataclass(frozen=True)
class RelationSpec:
    """Describe a named relation used by relational lokas."""

    name: str
    inputs: Tuple[str, ...]
    result: Union[str, Tuple[str, ...]]
    theory_refs: Tuple[str, ...]
    description: str = ""

    def normalized_key(self) -> Tuple[int, Tuple[str, ...]]:
        """Return a hashable key based on the sorted input names."""

        return len(self.inputs), tuple(sorted(self.inputs))


class SuperpositionalLoka(Loka):
    """Combine multiple lokas into a shared space while preserving Sigma balance."""

    def __init__(
        self,
        name: str,
        polarities: List[Polarity],
        *,
        layers: Sequence[SuperpositionLayer] | None = None,
        delegate_layer: str | SuperpositionLayer | None = None,
        tattva: TattvaProfile | str | None = None,
        dharmas: Sequence[DharmaProfile | str] | None = None,
        mind_modes: Sequence[str] | None = None,
        intensity_modes: Sequence[str] | None = None,
    ) -> None:
        super().__init__(
            name,
            polarities,
            tattva=tattva or "triadic_superposition",
            dharmas=dharmas,
            mind_modes=mind_modes,
            intensity_modes=intensity_modes,
        )

        self.layers = tuple(layers or ())
        self._layer_by_name: Dict[str, SuperpositionLayer] = {}
        self._layer_by_source: Dict[str, SuperpositionLayer] = {}
        self._layer_inverse: Dict[str, Dict[str, str]] = {}

        for layer in self.layers:
            norm_map = layer.normalized_mapping()
            if layer.name in self._layer_by_name:
                raise ValueError(f"duplicate superposition layer '{layer.name}'")
            self._layer_by_name[layer.name] = layer
            self._layer_by_source[layer.source_loka.name] = layer
            inverse: Dict[str, str] = {}
            for src_name, target_name in norm_map.items():
                if self.get_polarity_by_name(target_name) is None:
                    raise ValueError(
                        f"target polarity '{target_name}' is not present in loka '{self.name}'"
                    )
                inverse[target_name] = src_name
            self._layer_inverse[layer.name] = inverse

        self._delegate_layer = self._resolve_layer(delegate_layer) if (delegate_layer or self.layers) else None

        neutral_targets: List[str] = []
        for layer in self.layers:
            src_neutral = layer.source_loka.neutral_element
            if src_neutral is None:
                continue
            target_name = layer.normalized_mapping().get(src_neutral.name)
            if target_name is not None:
                neutral_targets.append(target_name)
        if neutral_targets:
            if len({target for target in neutral_targets}) > 1:
                raise ValueError("superposition layers must share a common neutral polarity")
            candidate = self.get_polarity_by_name(neutral_targets[0])
            if candidate is None:
                raise ValueError("neutral polarity from layer was not found in the resulting loka")
            self.neutral_element = candidate

    def _resolve_layer(self, layer: SuperpositionLayer | str | None) -> SuperpositionLayer | None:
        if layer is None:
            return self.layers[0] if self.layers else None
        if isinstance(layer, SuperpositionLayer):
            return layer
        if layer in self._layer_by_name:
            return self._layer_by_name[layer]
        if layer in self._layer_by_source:
            return self._layer_by_source[layer]
        raise KeyError(f"layer '{layer}' is not registered in {self.name}")

    def _delegate(self) -> SuperpositionLayer:
        layer = self._delegate_layer or self._resolve_layer(None)
        if layer is None:
            raise NotImplementedError(f"loka '{self.name}' has no delegate superposition layer")
        return layer

    def project_from_layer(
        self,
        value: MultipolarValue,
        *,
        layer: SuperpositionLayer | str | None = None,
    ) -> MultipolarValue:
        """Project a value from its source loka into this superposition."""

        if layer is None:
            spec = (
                self._layer_by_source.get(value.loka.name, None)
                if isinstance(getattr(value, "loka", None), Loka)
                else None
            )
            if spec is None:
                spec = self._resolve_layer(None)
        else:
            spec = self._resolve_layer(layer)
        if spec is None:
            raise ValueError("unable to determine superposition layer for projection")
        mapping = spec.normalized_mapping()
        projected: Dict[Polarity, complex] = {}
        for polarity, coeff in value.coefficients.items():
            target_name = mapping.get(polarity.name)
            if target_name is None:
                raise KeyError(
                    f"polarity '{polarity.name}' is not mapped from layer {spec.name} into {self.name}"
                )
            target_pol = self.get_polarity_by_name(target_name)
            if target_pol is None:
                raise KeyError(
                    f"target polarity '{target_name}' is not present in loka '{self.name}'"
                )
            projected[target_pol] = projected.get(target_pol, 0) + coeff
        return MultipolarValue(self, projected)

    def superpose_values(self, *values: MultipolarValue) -> MultipolarValue:
        """Sum several values by projecting them into this loka."""

        combined: Dict[Polarity, complex] = {}
        for value in values:
            projected = self.project_from_layer(value)
            for polarity, coeff in projected.coefficients.items():
                combined[polarity] = combined.get(polarity, 0) + coeff
        return MultipolarValue(self, combined)

    def _lift(self, polarity: Polarity, *, layer: SuperpositionLayer) -> Polarity:
        inverse = self._layer_inverse.get(layer.name, {})
        src_name = inverse.get(polarity.name)
        if src_name is None:
            raise KeyError(f"polarity '{polarity.name}' is not mapped in layer {layer.name}")
        src_pol = layer.source_loka.get_polarity_by_name(src_name)
        if src_pol is None:
            raise KeyError(
                f"polarity '{src_name}' is not present in source loka '{layer.source_loka.name}'"
            )
        return src_pol

    def multiply(self, p1: Polarity, p2: Polarity) -> Polarity:
        layer = self._delegate()
        src_p1 = self._lift(self._check_operands(p1, p1)[0], layer=layer)
        src_p2 = self._lift(self._check_operands(p2, p2)[0], layer=layer)
        product = layer.source_loka.multiply(src_p1, src_p2)
        target_name = layer.normalized_mapping().get(product.name)
        if target_name is None:
            raise RuntimeError(f"product {product.name} is not mapped back into {self.name}")
        target_pol = self.get_polarity_by_name(target_name)
        if target_pol is None:
            raise RuntimeError(f"polarity '{target_name}' is missing in loka '{self.name}'")
        return target_pol

    def add(self, p1: Polarity, p2: Polarity) -> Polarity:
        layer = self._delegate()
        src_p1 = self._lift(self._check_operands(p1, p1)[0], layer=layer)
        src_p2 = self._lift(self._check_operands(p2, p2)[0], layer=layer)
        result = layer.source_loka.add(src_p1, src_p2)
        target_name = layer.normalized_mapping().get(result.name)
        if target_name is None:
            raise RuntimeError(f"sum {result.name} is not mapped back into {self.name}")
        target_pol = self.get_polarity_by_name(target_name)
        if target_pol is None:
            raise RuntimeError(f"polarity '{target_name}' is missing in loka '{self.name}'")
        return target_pol

    def inverse(self, polarity: Polarity) -> Polarity:
        layer = self._delegate()
        src_pol = self._lift(self._check_operands(polarity, polarity)[0], layer=layer)
        inv = layer.source_loka.inverse(src_pol)
        target_name = layer.normalized_mapping().get(inv.name)
        if target_name is None:
            raise RuntimeError(f"inverse {inv.name} is not mapped back into {self.name}")
        target_pol = self.get_polarity_by_name(target_name)
        if target_pol is None:
            raise RuntimeError(f"polarity '{target_name}' is missing in loka '{self.name}'")
        return target_pol

    def divide(self, numerator: Polarity, denominator: Polarity) -> Polarity:
        layer = self._delegate()
        src_num = self._lift(self._check_operands(numerator, numerator)[0], layer=layer)
        src_den = self._lift(self._check_operands(denominator, denominator)[0], layer=layer)
        result = layer.source_loka.divide(src_num, src_den)
        target_name = layer.normalized_mapping().get(result.name)
        if target_name is None:
            raise RuntimeError(f"quotient {result.name} is not mapped back into {self.name}")
        target_pol = self.get_polarity_by_name(target_name)
        if target_pol is None:
            raise RuntimeError(f"polarity '{target_name}' is missing in loka '{self.name}'")
        return target_pol


class Harloka(SuperpositionalLoka):
    """Superpositional loka that wires fiber bundles into the resulting space."""

    def __init__(
        self,
        name: str,
        polarities: List[Polarity],
        *,
        bundles: Sequence[HarBundle] | None = None,
        layers: Sequence[SuperpositionLayer] | None = None,
        tattva: TattvaProfile | str | None = None,
        dharmas: Sequence[DharmaProfile | str] | None = None,
        mind_modes: Sequence[str] | None = None,
        intensity_modes: Sequence[str] | None = None,
    ) -> None:
        har_layers: List[SuperpositionLayer] = list(layers or [])
        for bundle in bundles or ():
            har_layers.append(
                SuperpositionLayer(
                    name=bundle.name,
                    source_loka=bundle.fiber_loka,
                    mapping=bundle.projection,
                    intensity_mode="sigma_add",
                    description=bundle.description,
                )
            )
        super().__init__(
            name,
            polarities,
            layers=har_layers,
            tattva=tattva or "relational_visibility",
            dharmas=dharmas,
            mind_modes=mind_modes,
            intensity_modes=intensity_modes,
        )
        self.bundles = tuple(bundles or ())


class RelationalLoka(Loka):
    """Relational loka that exposes visibility rules explicitly."""

    def __init__(
        self,
        name: str,
        polarities: List[Polarity],
        *,
        relations: Sequence[RelationSpec] | None = None,
        tattva: TattvaProfile | str | None = None,
        dharmas: Sequence[DharmaProfile | str] | None = None,
        mind_modes: Sequence[str] | None = None,
        intensity_modes: Sequence[str] | None = None,
    ) -> None:
        super().__init__(
            name,
            polarities,
            tattva=tattva or "relational_visibility",
            dharmas=dharmas,
            mind_modes=mind_modes,
            intensity_modes=intensity_modes,
        )
        self._relations: Dict[Tuple[int, Tuple[str, ...]], RelationSpec] = {}
        if relations:
            for relation in relations:
                self.add_relation(relation)

    def add_relation(self, relation: RelationSpec) -> None:
        if not isinstance(relation, RelationSpec):
            raise TypeError("relation must be an instance of RelationSpec")
        key = relation.normalized_key()
        existing = self._relations.get(key)
        if existing and existing != relation:
            raise ValueError(f"conflicting relation for inputs {relation.inputs} in '{self.name}'")
        for name in relation.inputs:
            if self.get_polarity_by_name(name) is None:
                raise ValueError(f"polarity '{name}' is not part of loka '{self.name}'")
        self._relations[key] = relation

    def list_relations(self) -> Tuple[RelationSpec, ...]:
        """Return all relations in a reproducible order."""

        return tuple(self._relations[key] for key in sorted(self._relations))

    def _resolve_result(self, spec: RelationSpec) -> Polarity | str:
        result = spec.result
        if isinstance(result, str):
            pol = self.get_polarity_by_name(result)
            if pol is not None:
                return pol
            if self.neutral_element and result == self.neutral_element.name:
                return self.neutral_element
            return result
        names = [self.get_polarity_by_name(name) for name in result]
        if all(name is not None for name in names):
            if len(names) == 1:
                return names[0]
            return "*".join(name.name for name in names if name is not None)
        return "*".join(result)

    def evaluate(self, polarities: List[Polarity]) -> Union[Polarity, str]:
        if not polarities:
            return "undefined relation (empty input)"
        internal: List[str] = []
        for polarity in polarities:
            pol_internal = self.get_polarity_by_name(polarity.name)
            if pol_internal is None:
                return "polarity does not belong to this loka"
            if self.neutral_element and pol_internal == self.neutral_element:
                continue
            internal.append(pol_internal.name)
        if not internal:
            return self.neutral_element or "undefined relation"
        key = (len(internal), tuple(sorted(internal)))
        spec = self._relations.get(key)
        if spec is None:
            return "undefined relation"
        return self._resolve_result(spec)

    def multiply(self, p1: Polarity, p2: Polarity) -> Polarity:
        p1_checked, p2_checked = self._check_operands(p1, p2)
        if self.neutral_element:
            if p1_checked == self.neutral_element:
                return p2_checked
            if p2_checked == self.neutral_element:
                return p1_checked
        result = self.evaluate([p1_checked, p2_checked])
        if isinstance(result, Polarity):
            return result
        raise RuntimeError(
            f"undefined multiplication {p1_checked.name}*{p2_checked.name} in {self.name}: {result}"
        )

    def inverse(self, polarity: Polarity) -> Polarity:
        p_checked, _ = self._check_operands(polarity, polarity)
        result = self.evaluate([p_checked, p_checked])
        if isinstance(result, Polarity) and (self.neutral_element is None or result == self.neutral_element):
            return p_checked
        if isinstance(result, Polarity):
            return result
        return p_checked

    def divide(self, numerator: Polarity, denominator: Polarity) -> Polarity:
        return self.multiply(numerator, self.inverse(denominator))


class LokaCn(Loka):
    """Cyclic loka C_n bound to theory-aligned tattva selection."""

    def __init__(self, n: int, operation_type: str, loka_name: str, polarity_names: List[str]):
        if not isinstance(n, int) or n < 1:
            raise ValueError("rank n must be a positive integer")
        if operation_type not in ["multiply", "add"]:
            raise ValueError("operation_type must be 'multiply' or 'add'")
        if len(polarity_names) != n:
            raise ValueError(
                f"number of polarity names ({len(polarity_names)}) must match rank n ({n})"
            )

        initialized_polarities: List[Polarity] = []
        for index, name in enumerate(polarity_names):
            value = cmath.exp(2 * cmath.pi * index * 1j / n)
            initialized_polarities.append(Polarity(name, value=value))

        tattva_map = {
            1: "unit_conserve",
            2: "binary_linear",
            3: "triadic_superposition",
            4: "complex_wave",
            5: "quaternion_harloka",
            6: "relational_visibility",
            7: "relational_visibility",
            8: "complex_wave",
            9: "quaternion_harloka",
            10: "relational_visibility",
        }
        tattva_name = tattva_map.get(n, "hyper_wave")

        super().__init__(
            name=loka_name,
            polarities=initialized_polarities,
            tattva=tattva_name,
        )
        self.n = n
        self.operation_type = operation_type
        self.cyclicity = n
        self._polarity_to_index = {polarity: idx for idx, polarity in enumerate(self.polarities)}
        self.neutral_element = self.polarities[0]

    def is_even_rank(self) -> bool:
        return (self.n % 2) == 0

    def is_odd_rank(self) -> bool:
        return (self.n % 2) == 1

    def get_order_two_element(self) -> Polarity | None:
        if not self.is_even_rank():
            return None
        return self.polarities[self.n // 2]

    def sum_full_set(self) -> Polarity:
        if self.operation_type != "add":
            raise NotImplementedError("sum_full_set() is only defined for additive lokas")
        current = self.polarities[0]
        for polarity in self.polarities[1:]:
            current = self.add(current, polarity)
        return current

    def product_full_set(self) -> Polarity:
        if self.operation_type != "multiply":
            raise NotImplementedError("product_full_set() is only defined for multiplicative lokas")
        current = self.polarities[0]
        for polarity in self.polarities[1:]:
            current = self.multiply(current, polarity)
        return current

    def _operate(self, p1: Polarity, p2: Polarity) -> Polarity:
        op1_internal, op2_internal = self._check_operands(p1, p2)
        idx1 = self._polarity_to_index[op1_internal]
        idx2 = self._polarity_to_index[op2_internal]
        return self.polarities[(idx1 + idx2) % self.n]

    def multiply(self, p1: Polarity, p2: Polarity) -> Polarity:
        if self.operation_type != "multiply":
            raise NotImplementedError(f"multiplication is not enabled for loka {self.name}")
        return self._operate(p1, p2)

    def add(self, p1: Polarity, p2: Polarity) -> Polarity:
        if self.operation_type != "add":
            raise NotImplementedError(f"addition is not enabled for loka {self.name}")
        return self._operate(p1, p2)

    def inverse(self, polarity: Polarity) -> Polarity:
        op_internal, _ = self._check_operands(polarity, polarity)
        idx = self._polarity_to_index[op_internal]
        return self.polarities[0] if idx == 0 else self.polarities[self.n - idx]

    def divide(self, numerator: Polarity, denominator: Polarity) -> Polarity:
        return self._operate(numerator, self.inverse(denominator))


class LokaDPN(Loka):
    """Loka implementing the complement principle (DPN)."""

    def __init__(
        self,
        n: int,
        *,
        enable_triple_neutral: bool = False,
        disable_self_interaction: bool = False,
    ) -> None:
        if n < 3:
            raise ValueError("LokaDPN is defined for n >= 3 non-neutral polarities")

        neutral = Polarity("O")
        polarities = [neutral] + [Polarity(chr(ord("A") + idx)) for idx in range(n)]
        super().__init__(
            name=f"LokaDP{n + 1}",
            polarities=polarities,
            tattva="relational_visibility",
        )
        self.neutral_element = neutral
        self.non_neutral_polarities = tuple(
            polarity for polarity in self.polarities if polarity != self.neutral_element
        )
        self.enable_triple_neutral = bool(enable_triple_neutral)
        self.disable_self_interaction = bool(disable_self_interaction)

    def evaluate(self, polarities: List[Polarity]) -> Union[Polarity, str]:
        if not polarities:
            return "undefined relation (empty input)"
        if len(polarities) != len({polarity.name for polarity in polarities}):
            if not self.disable_self_interaction:
                if (
                    len(polarities) == 2
                    and polarities[0] == polarities[1]
                    and polarities[0].name != "O"
                ):
                    return self.neutral_element

        eval_polarities = {self.get_polarity_by_name(polarity.name) for polarity in polarities}
        has_neutral = self.neutral_element in eval_polarities
        if has_neutral:
            eval_polarities.remove(self.neutral_element)
        if not eval_polarities:
            return self.neutral_element
        if len(eval_polarities) == 1 and has_neutral:
            return list(eval_polarities)[0]

        k = len(eval_polarities)
        total = len(self.non_neutral_polarities)

        if self.enable_triple_neutral and (total % 3 == 0) and k == 3:
            if all(polarity in self.non_neutral_polarities for polarity in eval_polarities):
                return self.neutral_element

        if k > total:
            return "undefined relation (too many polarities)"
        if k == total:
            return self.neutral_element

        complement_set = set(self.non_neutral_polarities) - eval_polarities
        if len(complement_set) == 1:
            return list(complement_set)[0]
        return "*".join(sorted(polarity.name for polarity in complement_set))

    def multiply(self, p1: Polarity, p2: Polarity) -> Polarity:
        p1_checked, p2_checked = self._check_operands(p1, p2)
        result = self.evaluate([p1_checked, p2_checked])
        if isinstance(result, Polarity):
            return result
        raise RuntimeError(
            f"undefined product {p1_checked.name}*{p2_checked.name} in {self.name}: {result}"
        )

    def inverse(self, polarity: Polarity) -> Polarity:
        p_checked, _ = self._check_operands(polarity, polarity)
        return p_checked

    def divide(self, numerator: Polarity, denominator: Polarity) -> Polarity:
        return self.multiply(numerator, self.inverse(denominator))


__all__ = [
    "SuperpositionLayer",
    "HarBundle",
    "RelationSpec",
    "SuperpositionalLoka",
    "Harloka",
    "RelationalLoka",
    "LokaCn",
    "LokaDPN",
]
