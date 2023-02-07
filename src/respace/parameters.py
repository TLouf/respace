from __future__ import annotations

import contextlib
from collections.abc import Hashable
from dataclasses import dataclass, field

import pandas as pd


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

    def __post_init__(self):
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
    parameters : list[Parameter] | dict
        Input list of `Parameter` instancess, or dictionary whose keys are parameter
        names, and whose values are either a single value, which will be the
        default, or a sequence of values, the first of which will be the default.

    Attributes
    ----------
    parameters : list[Parameter]
        Sorted list of `Parameter` instances.
    """

    def __init__(self, parameters: list[Parameter] | dict):
        if isinstance(parameters, dict):
            parameters = [
                Parameter(key, v)
                if pd.api.types.is_scalar(v)
                else Parameter(key, v[0], v)
                for key, v in parameters.items()
            ]

        self.parameters = sorted(parameters, key=lambda p: p.name)

    def __iter__(self):
        return iter(self.parameters)

    def to_dict(self) -> dict:
        """Return a dictionary giving the possible values of all parameters."""
        return {p.name: p.values for p in self}
