"""ReSpace."""
import sys

if sys.version_info < (3, 10):
    # compatibility for python <3.10
    import importlib_metadata as metadata
else:
    from importlib import metadata

from respace.parameters import Parameter, ParameterSet
from respace.result import ResultMetadata, ResultSet, ResultSetMetadata

__all__ = [
    "Parameter",
    "ParameterSet",
    "ResultMetadata",
    "ResultSet",
    "ResultSetMetadata",
]

__version__ = metadata.version("respace")
