from __future__ import annotations

import inspect
from collections.abc import Callable, Hashable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from typing_extensions import Self

from respace._typing import ParamsType
from respace.parameters import ParameterSet
from respace.utils import _tracking, save_pickle

xr.set_options(keep_attrs=True)


@dataclass
class ResultMetadata:
    name: str
    compute_fun: Callable
    save_fun: Callable = save_pickle
    save_suffix: str = ".pickle"


@dataclass
class ResultSetMetadata:
    results: list[ResultMetadata]

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        # Keys are names
        results = []
        for name, metadata in d.items():
            if not isinstance(metadata, dict):
                # Then metadata is `compute_fun`.
                metadata = {"compute_fun": metadata}
            results.append(ResultMetadata(name, **metadata))
        return cls(results)

    def __iter__(self) -> Iterator[ResultMetadata]:
        return iter(self.results)


class ResultSet:
    def __init__(
        self,
        results_metadata: ResultSetMetadata | dict,
        params: ParamsType,
        attrs: dict | None = None,
        save_path_fmt: str | Path | None = None,
    ) -> None:
        params_set = params
        if not isinstance(params_set, ParameterSet):
            params_set = ParameterSet(params_set)
        if isinstance(results_metadata, dict):
            results_metadata = ResultSetMetadata.from_dict(results_metadata)

        dims = [p.name for p in params_set]
        data = -np.ones([len(p.values) for p in params_set], dtype="int")
        data_vars = {
            r.name: xr.Variable(
                dims,
                data,
                attrs={
                    "tracking_compute_fun": _tracking(self, r.name)(r.compute_fun),
                    "computed_values": [],
                    "compute_times": [],
                    **asdict(r),
                },
            )  # type: ignore[no-untyped-call]
            for r in results_metadata
        }
        params_dict = params_set.to_dict()
        self.param_space = xr.Dataset(
            data_vars=data_vars,
            coords=params_set.to_dict(),
            attrs=attrs,
        )
        if save_path_fmt is None:
            save_path_fmt = "_".join(
                ["{res_name}"] + [f"{p}={{p}}" for p in params_dict]
            )
        if isinstance(save_path_fmt, str):
            save_path_fmt = Path(save_path_fmt)
        self.save_path_fmt = save_path_fmt

    def __str__(self) -> str:
        return str(self.param_space)

    def __repr__(self) -> str:
        return repr(self.param_space)

    def _repr_html_(self) -> str:
        return self.param_space._repr_html_()

    def __getitem__(self, r: Hashable | list[Hashable]):
        """Get the parameter space for one or a set of results.

        Parameters
        ----------
        r : str | list[str]
            Name of list of names of results to get the parameter space of.

        Returns
        -------
        param_space
            Parameter space(s) of `r`.
        """
        return self.param_space[r]

    @property
    def attrs(self):
        """Return the attributes dictionary of the ResultSet."""
        return self.param_space.attrs

    @attrs.setter
    def attrs(self, attrs_: dict) -> None:
        self.param_space.attrs = attrs_

    @property
    def coords(self):
        """Return the coordinates of the parameter space."""
        return self.param_space.coords

    @property
    def save_path_fmt(self):
        """Return the save path format of the results."""
        return self._save_path_fmt

    @save_path_fmt.setter
    def save_path_fmt(self, save_path_fmt: str | Path) -> None:
        self._save_path_fmt = Path(save_path_fmt)

    @property
    def params_values(self) -> dict[str, Sequence[Hashable]]:
        """Return a dictionary with the possible values of all parameters."""
        return {
            param_name: param_values.data
            for param_name, param_values in self.coords.items()
        }

    @property
    def param_defaults(self) -> dict[str, Hashable]:
        """Return a dictionary with the default values of all parameters."""
        d = {
            param_name: param_values.data[0]
            for param_name, param_values in self.coords.items()
        }
        return d

    @property
    def results_metadata(self) -> dict[str, Callable]:
        """Return a dictionary giving the metadata for all results."""
        metadata = {
            name: {
                attr_name: attr_value
                for attr_name, attr_value in self[name].attrs.items()
                if attr_name not in ("computed_values", "compute_times")
            }
            for name in self.param_space.data_vars
        }
        return metadata

    def set_compute_fun(self, res_name: str, compute_fun: Callable):
        """Set the computing funtion for `res_name`.

        Parameters
        ----------
        res_name : str
            Name of the result for which to set the computing function.
        compute_fun : Callable
            New computing function of `res_name` to set.
        """
        res_attrs = self[res_name].attrs
        res_attrs["compute_fun"] = compute_fun
        res_attrs["tracking_compute_fun"] = _tracking(self, res_name)(compute_fun)

    def set_save_fun(self, res_name: str, save_fun: Callable):
        """Set the computing funtion for `res_name`.

        Parameters
        ----------
        res_name : str
            Name of the result for which to set the computing function.
        save_fun : Callable
            New saving function of `res_name` to set, taking the result instance and the
            save path as respectively first and second arguments.
        """
        self[res_name].attrs["save_fun"] = save_fun

    def fill_with_defaults(self, params: dict) -> dict:
        return {**self.param_defaults, **params}

    def is_computed(self, res_name, params: dict) -> bool:
        complete_param_set = self.fill_with_defaults(params)
        return self[res_name].loc[complete_param_set] >= 0

    def compute(self, res_name, params: dict, **add_kwargs):
        complete_param_set = self.fill_with_defaults(params)
        # Add other results to add_kwargs if necessary.
        argspec = inspect.getfullargspec(self[res_name].compute_fun)
        possible_kwds = set(argspec.args + argspec.kwonlyargs)
        res_names = set(self.param_space.data_vars.keys())
        other_res_deps = {
            rn: self.get(rn, complete_param_set, **add_kwargs)
            for rn in possible_kwds.intersection(res_names)
        }
        add_kwargs = {**add_kwargs, **other_res_deps}
        try:
            result = self[res_name].tracking_compute_fun(
                **complete_param_set, **add_kwargs
            )
        except TypeError as e:
            raise e(
                "You passed new parameters, but it seems `compute_fun` was not updated"
                " to accept them."
            ) from e

        # TODO: avoid if unnecessary?
        self.add_param_values(complete_param_set)
        res_coord = len(self[res_name].attrs["computed_values"]) - 1
        self[res_name].loc[complete_param_set] = res_coord
        return result

    def get(self, res_name: str, params: dict, **add_kwargs):
        """Get the value of result `res_name` for set of parameters `params`.

        Parameters
        ----------
        res_name : str
            Name of the result to get.
        params : dict
            Dictionary of parameters for which to get the result.
        **add_kwargs:
            Additional keyword arguments to pass to `res_name`'s computing function,
            like external data for instance.

        Returns
        -------
        result value
        """
        complete_param_set = self.fill_with_defaults(params)
        try:
            res_coord = self[res_name].loc[complete_param_set].values
            if res_coord >= 0:
                return self[res_name].attrs["computed_values"][res_coord]
        except KeyError:
            # means a new parameter was passed from params, will be added by `compute`
            # anyway
            pass

        return self.compute(res_name, complete_param_set, **add_kwargs)

    def save(
        self,
        res_name: str,
        params: dict,
        save_path_fmt: Path | str | None = None,
        **add_kwargs,
    ) -> Any:
        save_fun = self[res_name].attrs["save_fun"]
        save_path = self.get_save_path(res_name, params, save_path_fmt=save_path_fmt)
        res_value = self.get(res_name, params, **add_kwargs)
        save_fun(res_value, save_path)
        return res_value

    def get_save_path(
        self, res_name: str, params: dict, save_path_fmt: Path | str | None = None
    ) -> Path:
        if save_path_fmt is None:
            save_path_fmt = str(self.save_path_fmt)
        else:
            save_path_fmt = str(save_path_fmt)

        complete_param_set = self.fill_with_defaults(params)
        save_path = Path(save_path_fmt.format(res_name=res_name, **complete_param_set))
        save_suffix = self[res_name].attrs["save_suffix"]
        return save_path.with_suffix(save_suffix)

    def get_nth_last_computed(self, res_name: str, n: int = 1):
        return self[res_name].attrs["computed_values"][-n]

    def get_nth_last_details(self, res_name: str, n: int = 1):
        value = self.get_nth_last_computed(res_name, n=n)
        params_idc = np.nonzero(
            self[res_name].param_space.data
            == len(self[res_name].attrs["computed_values"]) - n
        )
        params = {}
        for i, d in enumerate(self.param_space.coords.keys()):
            params[d] = self.coords[d].data[params_idc[i]][0]
        return value, params

    def add_param_values(self, values: dict):
        self.param_space = self.param_space.reindex(
            {p: np.union1d(v, self.coords[p].values) for p, v in values.items()},
            fill_value=-1,
        ).copy()

    def add_params(self, params: ParamsType):
        params_set = params
        if not isinstance(params_set, ParameterSet):
            params_set = ParameterSet(params_set)

        # always add so that .dims is ordered!
        curr_dims = list(self.param_space.coords.keys())
        add_dims = np.asarray([p.name for p in params_set])
        add_dims_sorting = np.argsort(add_dims)
        # this works because curr_dims is assumed always sorted
        axis_of_sorted_added_dims = np.searchsorted(
            curr_dims, add_dims[add_dims_sorting]
        ) + np.arange(add_dims.size)
        axis = axis_of_sorted_added_dims[add_dims_sorting].tolist()
        params_dict = params_set.to_dict()
        self.param_space = self.param_space.expand_dims(params_dict, axis=axis)

    def populated_mask(self):
        return self.param_space >= 0

    def populated_space(self):
        return np.where(self.populated_mask, drop=True)

    def get_subspace_res(
        self, subspace_params: dict, keep_others_default_only: bool = False
    ) -> ResultSet:
        """_summary_.

        TODO: get subset of res with all this functionality too

        Parameters
        ----------
        subspace_params : dict
            _description_
        keep_others_default_only : bool, optional
            If False, the default, for parameters not specified in `subspace_params`,
            keep only the coordinate of their defaults values. If True, keep all their
            values.

        Returns
        -------
        Result
            _description_
        """
        if keep_others_default_only:
            complete_param_set = self.fill_with_defaults(subspace_params)
        else:
            complete_param_set = {**self.params_values, **subspace_params}
        param_subspace = self.param_space.loc[complete_param_set]
        subspace_res = ResultSet(
            compute_dict=self.compute_dict, params=complete_param_set
        )
        # values from ranking of flattened values -> array of ranks. Then subsitute
        # number of -1s in original array to have 0 actually correspond to the first
        # element.
        for res_name in param_subspace.data_vars:
            param_data = (
                param_subspace[res_name]
                .data.flatten()
                .argsort()
                .argsort()
                .reshape(param_subspace.shape)
            )
            param_data = param_data - (param_subspace[res_name] < 0).sum()
            subspace_res.param_space[res_name].data = param_data
            idc_res = np.unique(param_subspace[res_name])
            subspace_res[res_name].attrs["computed_values"] = [
                self[res_name].attrs["computed_values"][i] for i in idc_res[1:]
            ]
            subspace_res[res_name].attrs["compute_times"] = [
                self[res_name].attrs["compute_times"][i] for i in idc_res[1:]
            ]
        return subspace_res
