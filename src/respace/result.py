from __future__ import annotations

from collections.abc import Callable

import numpy as np
import xarray as xr

from respace._typing import ParamsType
from respace.parameters import ParameterSet
from respace.utils import _tracking

xr.set_options(keep_attrs=True)


# TODO: handle results inter-dependence. check in params of compute_fun, if name matches
# another result's, compute it and add to kwargs dictionary
class ResultSet:
    def __init__(
        self,
        compute_dict: dict[str, Callable],
        params: ParamsType,
    ):
        """Init.

        param_space
            dataarray which gives index of `values` where to find the result of
            computation with set of parameters corresponding to coordinates of array.
            The default value of -1 means it has not been computed, and a new value
            shall be computed and appended to `values`. by definitions then, parameters
            need be of consistent type, and be Hashable (because DataArray coordinates
            are based on pandas.Index (ref
            https://docs.xarray.dev/en/stable/user-guide/terminology.html#term-Dimension-coordinate)
            and a pandas Index can only contain hashable objects (ref
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Index.html#pandas.Index)
        values
            this is list so it can hold any type (including mixed ones).
        """
        params_set = params
        if not isinstance(params_set, ParameterSet):
            params_set = ParameterSet(params_set)
        dims = [p.name for p in params_set]
        data = -np.ones([len(p.values) for p in params_set], dtype=int)
        data_vars = {
            n: xr.Variable(
                dims,
                data,
                attrs={
                    "compute_fun": fun,
                    "tracking_compute_fun": _tracking(self, n)(fun),
                    "computed_values": [],
                    "compute_times": [],
                },
            )
            for n, fun in compute_dict.items()
        }
        self.param_space = xr.Dataset(
            data_vars=data_vars,
            coords=params_set.to_dict(),
        )

    def __getitem__(self, r: str | list[str]):
        return self.param_space[r]

    @property
    def name(self):
        return self.param_space.name

    @name.setter
    def name(self, name_: str):
        self.param_space.name = name_

    @property
    def attrs(self):
        return self.param_space.attrs

    @attrs.setter
    def attrs(self, attrs_: dict):
        self.param_space.attrs = attrs_

    @property
    def coords(self):
        return self.param_space.coords

    @coords.setter
    def coords(self, coords_):
        self.param_space.coords = coords_

    @property
    def params_values(self) -> dict:
        return {
            param_name: param_values.data
            for param_name, param_values in self.coords.items()
        }

    @property
    def param_defaults(self) -> dict:
        d = {
            param_name: param_values.data[0]
            for param_name, param_values in self.coords.items()
            if param_values.size > 0
        }
        return d

    def fill_with_defaults(self, params: dict) -> dict:
        return {**self.param_defaults, **params}

    def is_computed(self, res_name, params: dict) -> bool:
        complete_param_set = self.fill_with_defaults(params)
        return self[res_name].loc[complete_param_set] >= 0

    def compute(self, res_name, params: dict, **add_kwargs):
        complete_param_set = self.fill_with_defaults(params)
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

    def get(self, res_name, params: dict, **add_kwargs):
        """Get.

        add_kwargs will hold other arguments needed for compute_fun that are not
        considered as params (external data eg).
        TODO: add **params_kwargs using this?
        https://github.com/pydata/xarray/blob/d385e2063a6b5919e1fe9dd3e27a24bc7117137e/xarray/core/utils.py#L270
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

        return self.compute(complete_param_set, **add_kwargs)

    def get_nth_last_computed(self, res_name, n=1):
        return self[res_name].attrs["computed_values"][-n]

    def get_nth_last_details(self, res_name, n=1):
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

    @property
    def compute_dict(self) -> dict[str, Callable]:
        return {
            name: self[name].attrs["compute_fun"] for name in self.param_space.data_vars
        }

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
