from __future__ import annotations

import inspect
import pickle
import time
from collections.abc import Hashable
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from respace.result import ResultSet


def save_pickle(object: Any, save_path: Path):
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


def _tracking(
    result_set: ResultSet,
    res_name: Hashable,
    timed: bool = True,
    append_values: bool = True,
):
    """Decorate the function to compute a result in a `ResultSet` to track its outputs.

    Parameters
    ----------
    result_set : ResultSet
        The :py:class:`respace.result.ResultSet` instance whose result `res_name` should
        have its `compute_fun` tracked.
    res_name : str
        Name of the result in `result_set` for whom the computing function should be
        made tracking.
    timed : bool, optional
        Whether to record the computing times in the `compute_times` attribute, by
        default True.
    append_values : bool, optional
        Whether to record the outputs in the `computed_values` attribute, by default
        True.
    """

    def decorator_compute(compute_fun):
        @wraps(compute_fun)
        def wrapper_compute(*args, **kwargs):
            argspec = inspect.getfullargspec(result_set[res_name].compute_fun)
            possible_kwds = argspec.args + argspec.kwonlyargs
            fun_kwargs = {
                kw: value for kw, value in kwargs.items() if kw in possible_kwds
            }
            if timed:
                start = time.time()
            result = compute_fun(*args, **fun_kwargs)
            if timed:
                end = time.time()
                result_set[res_name].attrs["compute_times"].append(end - start)
            if append_values:
                result_set[res_name].attrs["computed_values"].append(result)
            return result

        return wrapper_compute

    return decorator_compute
