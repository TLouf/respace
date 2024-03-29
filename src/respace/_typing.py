from __future__ import annotations

from collections.abc import Hashable
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Mapping,
    Sequence,
    TypedDict,
    Union,
)

if TYPE_CHECKING:
    from typing_extensions import Required

    from respace.parameters import Parameter, ParameterSet
    from respace.result import ResultMetadata

ParamsSingleValue = Mapping[str, Hashable]
ParamsMultValues = Mapping[str, Sequence[Hashable]]
ParamsArgType = Mapping[str, Union[Hashable, Sequence[Hashable]]]
ParamsType = Union["Parameter", List["Parameter"], ParamsArgType, "ParameterSet"]

ComputeFunType = Callable[..., Any]
SaveFunType = Callable[[Any, Union[Path, str]], Any]
LoadFunType = Callable[[Union[Path, str]], Any]


class ResultSetDict(TypedDict, total=False):
    compute_fun: Required[ComputeFunType]
    save_fun: SaveFunType
    load_fun: LoadFunType
    save_suffix: str
    save_path_fmt: str


ResultSetMetadataInput = Union[
    "ResultMetadata",
    List["ResultMetadata"],
    Mapping[str, Union[ComputeFunType, ResultSetDict]],
]
