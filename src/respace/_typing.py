from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from pathlib import Path

# Use of Union for rendering in Sphinx autodata directives
from typing import TYPE_CHECKING, Any, TypedDict, Union

from typing_extensions import Required

if TYPE_CHECKING:
    from respace.parameters import Parameter, ParameterSet

ParamsSingleValue = Mapping[str, Hashable]
ParamsMultValues = Mapping[str, Sequence[Hashable]]
ParamsArgType = Mapping[str, Hashable | Sequence[Hashable]]
ParamsType = Union[list["Parameter"], ParamsArgType, "ParameterSet"]

ComputeFunType = Callable[..., Any]
SaveFunType = Union[
    Callable[[Any, Path], None],
    Callable[[Any, str], None],
    Callable[[Any, Path | str], None],
]  # TODO: better way to do this?


class ResultSetDict(TypedDict, total=False):
    compute_fun: Required[ComputeFunType]
    save_fun: SaveFunType
    save_suffix: str
    save_path_fmt: str


ResultsMetadataDictType = Mapping[str, ComputeFunType | ResultSetDict]
