from __future__ import annotations

import contextlib
from collections.abc import Hashable
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Parameter:
    name: str
    default: Hashable
    values: list[Hashable] = field(default_factory=list)

    def __post_init__(self):
        # put default in self.values[0]
        if len(self.values) == 0:
            self.values.append(self.default)
        elif self.values[0] != self.default:
            with contextlib.suppress(ValueError):
                self.values.remove(self.default)
            self.values = [self.default] + self.values

    def to_dict(self):
        return {self.name: self.values}


class ParameterSet:
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

    def to_dict(self):
        return {p.name: p.values for p in self}
