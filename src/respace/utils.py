from __future__ import annotations

import inspect
import pickle
import sys
import time
from functools import wraps
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import TypeVar

    if sys.version_info < (3, 10):
        from typing_extensions import Concatenate
    else:
        from typing import Concatenate

    from typing_extensions import ParamSpec

    from respace.result import ResultSet

    R = TypeVar("R")
    P = ParamSpec("P")


def save_pickle(object: Any, save_path: Path | str) -> None:
    """Save `object` to pickle format at `save_path`.

    Parameters
    ----------
    object : Any
        Oject to save.
    save_path : Path
        Path where to save `object`.
    """
    with open(save_path, "wb") as f:
        pickle.dump(object, f)


def load_pickle(save_path: Path | str) -> Any:
    """Load `object` from file in pickle format at `save_path`.

    Parameters
    ----------
    save_path : Path
        Path where to load the object from.

    Returns
    -------
    Any
    """
    with open(save_path, "rb") as f:
        object = pickle.load(f)
    return object


def _tracking(
    result_set: ResultSet,
    res_name: str,
    timed: bool = True,
    append_values: bool = True,
) -> Callable[[Callable[P, R]], Callable[Concatenate[Any, P], R]]:
    """Decorate the function to compute a result in a `ResultSet` to track its outputs.

    Parameters
    ----------
    result_set : ResultSet
        The :py:class:`respace.result.ResultSet` instance whose result `res_name` should
        have its `compute_fun` tracked.
    res_name : str
        Name of the result in `result_set` for whom the computing function should be
        made tracking.
    timed : bool
        Whether to record the computing times in the `compute_times` attribute, by
        default True.
    append_values : bool
        Whether to record the outputs in the `computed_values` attribute, by default
        True.

    Returns
    -------
    Callable[[Callable[P, R]], Callable[Concatenate[Any, P], R]]
        Decorator for a computing function.
    """

    def decorator_compute(
        compute_fun: Callable[P, R]
    ) -> Callable[Concatenate[Any, P], R]:
        @wraps(compute_fun)
        # TODO: way to annotate type of kwargs as superset of P.kwargs? to remove
        # type ignore below.
        def wrapper_compute(*args: P.args, **kwargs: P.kwargs) -> R:
            argspec = inspect.getfullargspec(result_set[res_name].compute_fun)
            possible_kwds = argspec.args + argspec.kwonlyargs
            fun_kwargs = {
                kw: value for kw, value in kwargs.items() if kw in possible_kwds
            }
            if timed:
                start = time.time()
            result = compute_fun(*args, **fun_kwargs)  # type: ignore[arg-type]
            if timed:
                end = time.time()
                result_set[res_name].attrs["compute_times"].append(end - start)
            if append_values:
                result_set[res_name].attrs["computed_values"].append(result)
            return result

        return wrapper_compute

    return decorator_compute
