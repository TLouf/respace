from __future__ import annotations

from collections.abc import Hashable, Sequence

from respace.parameters import Parameter, ParameterSet

ParamsDictType = dict[str, Hashable | Sequence[Hashable]]
ParamsType = list[Parameter] | ParamsDictType | ParameterSet
