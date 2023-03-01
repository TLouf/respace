from __future__ import annotations

import contextlib
from collections.abc import Hashable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field

from respace._typing import ParamsArgType


@dataclass
class Parameter:
    """Represent a parameter with a mandatory default value.

    Attributes
    ----------
    name : str
        Name of the parameter.
    default : Hashable
        Default value for this parameter.
    values : list[Hashable]
        List of possible values for the parameter. Will contain the default in first
        position even if it was not supplied thus at initialization.
    """

    name: str
    default: Hashable
    values: list[Hashable] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Put default in self.values[0].
        if len(self.values) == 0:
            self.values.append(self.default)
        elif self.values[0] != self.default:
            with contextlib.suppress(ValueError):
                self.values.remove(self.default)
            self.values = [self.default] + self.values


class ParameterSet:
    """Hold a list of ``Parameter`` instances and facilitate iteration over them.

    The ``Parameter`` instances will be sorted according to their `name` attribute.

    Parameters
    ----------
    parameters : Parameter | list[Parameter] | ParamsArgType
        Input :class:`respace.Parameter` or list of `Parameter` instances, or dictionary
        whose keys are parameter names, and whose values are either a single value,
        which will be the default, or a sequence of values, the first of which will be
        the default.

    Attributes
    ----------
    parameters : list[Parameter]
        Sorted list of `Parameter` instances.
    """

    def __init__(
        self, parameters: Parameter | Iterable[Parameter] | ParamsArgType
    ) -> None:
        if isinstance(parameters, Mapping):
            parameters = [
                Parameter(key, v)
                if isinstance(v, Hashable)
                else Parameter(key, v[0], v)
                for key, v in parameters.items()
            ]
        elif isinstance(parameters, Parameter):
            parameters = [parameters]
        self.parameters = sorted(parameters, key=lambda p: p.name)

    def __iter__(self) -> Iterator[Parameter]:
        return iter(self.parameters)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ParameterSet):
            return self.parameters == other.parameters
        return NotImplemented

    def __getitem__(self, i: int) -> Parameter:
        return self.parameters[i]

    def to_dict(self) -> dict[str, list[Hashable]]:
        """Return a dictionary giving the possible values of all parameters."""
        return {p.name: p.values for p in self}
