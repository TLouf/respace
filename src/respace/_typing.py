from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from pathlib import Path

# Use of Union for rendering in Sphinx autodata directives
from typing import TYPE_CHECKING, Any, TypedDict, Union

from typing_extensions import Required

if TYPE_CHECKING:
    from respace.parameters import Parameter, ParameterSet
    from respace.result import ResultMetadata

ParamsSingleValue = Mapping[str, Hashable]
ParamsMultValues = Mapping[str, Sequence[Hashable]]
ParamsArgType = Mapping[str, Hashable | Sequence[Hashable]]
ParamsType = Union["Parameter", list["Parameter"], ParamsArgType, "ParameterSet"]

ComputeFunType = Callable[..., Any]
SaveFunType = Callable[[Any, Path | str], Any]
LoadFunType = Callable[[Path | str], Any]


class ResultSetDict(TypedDict, total=False):
    compute_fun: Required[ComputeFunType]
    save_fun: SaveFunType
    load_fun: LoadFunType
    save_suffix: str
    save_path_fmt: str


ResultSetMetadataInput = Union[
    "ResultMetadata",
    list["ResultMetadata"],
    Mapping[str, ComputeFunType | ResultSetDict],
]
