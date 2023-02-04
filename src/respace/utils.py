import inspect
import time
from functools import wraps


def _tracking(obj, res_name, timed: bool = True, append_values: bool = True):
    def decorator_compute(compute_fun):
        @wraps(compute_fun)
        def wrapper_compute(*args, **kwargs):
            # argspec = inspect.getfullargspec(compute_fun)
            argspec = inspect.getfullargspec(obj[res_name].compute_fun)
            posible_kwds = argspec.args + argspec.kwonlyargs
            fun_kwargs = {
                kw: value for kw, value in kwargs.items() if kw in posible_kwds
            }
            if timed:
                start = time.time()
            result = compute_fun(*args, **fun_kwargs)
            if timed:
                end = time.time()
                obj[res_name].attrs["compute_times"].append(end - start)
                # self.compute_times.append(end - start)
            if append_values:
                obj[res_name].attrs["computed_values"].append(result)
                # self.values.append(result)
            return result

        return wrapper_compute

    return decorator_compute
