from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict

from typing_extensions import Required

from respace.parameters import Parameter, ParameterSet

ParamsArgType = Mapping[str, Hashable | Sequence[Hashable]]
ParamsSingleValue = Mapping[str, Hashable]
ParamsMultValues = Mapping[str, Sequence[Hashable]]
ParamsType = list[Parameter] | ParamsArgType | ParameterSet

ComputeFunType = Callable[..., Any]
SaveFunType = (
    Callable[[Any, Path], None]
    | Callable[[Any, str], None]
    | Callable[[Any, Path | str], None]
)  # TODO: better way to do this?


class ResultSetDict(TypedDict, total=False):
    compute_fun: Required[str]
    save_fun: SaveFunType
    save_suffix: str
