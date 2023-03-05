from __future__ import annotations

import inspect
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

import numpy as np
import pandas as pd
import xarray as xr

# Has to be outside of `if TYPE_CHECKING` for sphinx autodoc to pick them up
from respace._typing import (
    ComputeFunType,
    LoadFunType,
    ParamsArgType,
    ParamsMultValues,
    ParamsSingleValue,
    ParamsType,
    ResultSetMetadataInput,
    SaveFunType,
)
from respace.parameters import Parameter, ParameterSet
from respace.utils import _tracking, load_pickle, save_pickle

if TYPE_CHECKING:
    import numpy.typing as npt
    from xarray.core.coordinates import DatasetCoordinates

xr.set_options(keep_attrs=True)  # type: ignore[no-untyped-call]


@dataclass
class ResultMetadata:
    """Represent a result with a mandatory name and computing function.

    Attributes
    ----------
    name : str
        Name of the result.
    compute_fun : ComputeFunType
        Function to call to compute the result.
    save_fun : SaveFunType, default :func:`respace.save_pickle`
        Function to call to save the result. Its first argument should be the value of the result to save, and the second the path where to save it.
    save_suffix : str
        Suffix of the file where the result would be saved. Should be set to match what's expected by to `save_fun`. Default is ".pickle"
    save_path_fmt : str, optional
        Default format for the path where to save the result. Will be formatted with the
        :meth:`str.format` method. If not set, :class:`respace.ResultSet`'s default will
        be used. Good practice is to include the name of the result and the parameters'
        with format fields for their values.
    """

    name: str
    compute_fun: ComputeFunType
    save_fun: SaveFunType = save_pickle
    load_fun: LoadFunType = load_pickle
    save_suffix: str = ".pickle"
    save_path_fmt: str | None = None


class ResultSetMetadata:
    """Represent a set of results with a mandatory name and computing function.

    Parameters
    ----------
    results : ResultSetMetadataInput
        Input :class:`respace.Parameter` or list of :class:`respace.Parameter`
        instances, or dictionary whose keys are result names, and whose values are
        either a computing function, or a dictionary matching the keys of
        :class:`respace.ResultMetadata`.

    Attributes
    ----------
    results : list[ResultMetadata]
        List of :class:`respace.ResultMetadata` instances.
    """

    def __init__(self, results: ResultSetMetadataInput) -> None:
        if isinstance(results, Mapping):
            results = [
                ResultMetadata(name, metadata)
                if callable(metadata)
                else ResultMetadata(name, **metadata)
                for name, metadata in results.items()
            ]
        elif isinstance(results, ResultMetadata):
            results = [results]
        self.results = results

    def __iter__(self) -> Iterator[ResultMetadata]:
        return iter(self.results)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ResultSetMetadata):
            return self.results == other.results
        return NotImplemented

    def __getitem__(self, i: int) -> ResultMetadata:
        return self.results[i]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.results)})"

    def copy(self) -> ResultSetMetadata:
        return ResultSetMetadata(self.results.copy())


# TODO: does allowing method chaining and not inplace operations make sense?
class ResultSet:
    """Hold a set of results within their parameter space.

    Parameters
    ----------
    results_metadata : ResultSetMetadata | ResultSetMetadataInput
        :class:`ResultSetMetadata`, (list of) :class:`ResultMetadata` or dictionary
        describing the results to add. See :class:`ResultSetMetadata` for more
        information on the dictionary format.
    params : ParamsType
        :class:`ParameterSet`, (list of) :class:`Parameter` or dictionary describing the
        parameters the results depend on. See :class:`ParameterSet` for more information
        on the dictionary format. They will be used as the coordinates of a
        :class:`xarray.Dataset` to keep track of the results' computed values. By
        definition then, each parameter should have a consistent type and be Hashable,
        because Dataset coordinates are based on :class:`pandas.Index`.
    attrs : dict, optional
        Global attributes to save on this result set.
    save_path_fmt : str | Path, optional
        Default format for the path where to save the results. Will be formatted with the
        :meth:`str.format` method, with a dictionary mapping "res_name" and the names of
        all parameters in the set to respectively the name of the result being saved and
        the value of the parameters. The default is
        ``"{res_name}_parameter1={parameter1}_ ..."``.
    verbose : bool
        Whether to print whenever a result is computed or saved.

    Attributes
    ----------
    save_path_fmt : str
        Default format for the path where to save the results.
    """

    def __init__(
        self,
        results_metadata: ResultSetMetadata | ResultSetMetadataInput,
        params: ParamsType,
        attrs: dict[str, Any] | None = None,
        save_path_fmt: str | Path | None = None,
        verbose: bool = False,
    ) -> None:
        params_set = params
        if not isinstance(params_set, ParameterSet):
            params_set = ParameterSet(params_set)
        if not isinstance(results_metadata, ResultSetMetadata):
            results_metadata = ResultSetMetadata(results_metadata)

        dims = [p.name for p in params_set]
        data = -np.ones([len(p.values) for p in params_set], dtype="int")
        data_vars = {
            r.name: self._make_res_var(r, dims, data) for r in results_metadata
        }
        self.param_space = xr.Dataset(
            data_vars=data_vars,
            coords=params_set.to_dict(),
            attrs=attrs,
        )
        self._save_path_fmt = save_path_fmt
        self.verbose = verbose

    def __str__(self) -> str:
        return str(self.param_space)

    def __repr__(self) -> str:
        return repr(self.param_space)

    def _repr_html_(self) -> str:
        return self.param_space._repr_html_()

    @overload
    def __getitem__(self, r: Hashable) -> xr.DataArray:
        ...

    @overload
    def __getitem__(self, r: list[Hashable]) -> xr.Dataset:
        ...

    def __getitem__(self, r: Hashable | list[Hashable]) -> xr.DataArray | xr.Dataset:
        """Get the parameter space for one or a set of the results.

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
    def _verbose_print(self) -> Callable[..., None]:
        if self.verbose:
            return print
        else:
            return lambda *a, **k: None

    @property
    def param_space(self) -> xr.Dataset:
        """Parameter space within which the results are computed.

        Returns
        -------
        param_space : xr.Dataset
            :class:`xarray.Dataset` which contains a :class:`xarray.DataArray` for each
            result of the set. Its coordinates are the possible values of all
            parameters. Its values are the indices where the result's computed values
            are stored in the list "computed_values", which is accessible as one of the
            attributes of the DataArray. The default value of -1 means the result has
            not been computed for the corresponding set of parameters, and a new value
            shall be computed and appended to "computed_values". The dimensions (one per
            parameter) are ordered by parameter name, in order to ensure invariance of
            ResultSets under different orders in adding parameters.
        """
        return self.__param_space

    @param_space.setter
    def param_space(self, param_space_: xr.Dataset) -> None:
        """Setter for the parameter space.

        Parameters
        ----------
        param_space_ : xr.Dataset
            Parameter space to set.

        Raises
        ------
        ValueError
            If ``param_space_.data_vars`` contains a variable whose name is not a
            string.
        ValueError
            If ``param_space_.coords`` contains a coordinate whose name is not a string.
        """
        for res_label in param_space_.data_vars:
            if not isinstance(res_label, str):
                raise ValueError("Result names should be strings.")

        for param_label in param_space_.coords:
            if not isinstance(param_label, str):
                raise ValueError("Parameter names should be strings")
        self.__param_space = param_space_

    @property
    def parameters(self) -> ParameterSet:
        # Type ignore below because don't know how to tell mypy we've locked coords so
        # that parameter labels as returned here below can only be strings.
        return ParameterSet(
            [
                Parameter(k, v.values[0], v.values.tolist())  # type: ignore[arg-type]
                for k, v in self.coords.items()
            ]
        )

    @property
    def params_values(self) -> ParamsMultValues:
        """Return a dictionary with the possible values of all parameters."""
        return {param.name: param.values for param in self.parameters}

    @property
    def params_defaults(self) -> ParamsSingleValue:
        """Return a dictionary with the default values of all parameters."""
        return {param.name: param.default for param in self.parameters}

    @property
    def results(self) -> ResultSetMetadata:
        return ResultSetMetadata(
            [
                ResultMetadata(
                    **{
                        attr_name: attr_value
                        for attr_name, attr_value in res.attrs.items()
                        if attr_name in ResultMetadata.__dataclass_fields__
                    }
                )
                for res in self.param_space.values()
            ]
        )

    @property
    def attrs(self) -> dict[str, Any]:
        """Return the attributes dictionary of the ResultSet."""
        return self.param_space.attrs

    @attrs.setter
    def attrs(self, attrs_: dict[str, Any]) -> None:
        self.param_space.attrs = attrs_

    @property
    def coords(self) -> DatasetCoordinates:
        """Return the coordinates of the parameter space."""
        return self.param_space.coords

    @property
    def save_path_fmt(self) -> str:
        """Return the default format for the path where to save the results."""
        if self._save_path_fmt is None:
            save_path_fmt_ = "_".join(
                ["{res_name}"] + [f"{p.name}={{{p.name}}}" for p in self.parameters]
            )
        else:
            save_path_fmt_ = str(self._save_path_fmt)
        return save_path_fmt_

    @save_path_fmt.setter
    def save_path_fmt(self, save_path_fmt: str | Path | None) -> None:
        self._save_path_fmt = save_path_fmt

    def set_compute_fun(self, res_name: str, compute_fun: ComputeFunType) -> None:
        """Set the computing funtion for `res_name`.

        Parameters
        ----------
        res_name : str
            Name of the result for which to set the computing function.
        compute_fun : ComputeFunType
            New computing function of `res_name` to set.
        """
        res_attrs = self[res_name].attrs
        res_attrs["compute_fun"] = compute_fun
        res_attrs["tracking_compute_fun"] = _tracking(self, res_name)(compute_fun)

    def set_save_fun(self, res_name: str, save_fun: SaveFunType) -> None:
        """Set the computing funtion for `res_name`.

        Parameters
        ----------
        res_name : str
            Name of the result for which to set the computing function.
        save_fun : SaveFunType
            New saving function of `res_name` to set, taking the result instance and the
            save path as respectively first and second arguments.
        """
        self[res_name].attrs["save_fun"] = save_fun

    @overload
    def fill_with_defaults(self, params: ParamsMultValues) -> ParamsMultValues:
        ...

    @overload
    def fill_with_defaults(self, params: ParamsSingleValue) -> ParamsSingleValue:
        ...

    def fill_with_defaults(self, params: ParamsArgType) -> ParamsArgType:
        """Fill `params` with the default values of the unspecified parameters."""
        return {**self.params_defaults, **params}

    def is_computed(self, res_name: str, params: ParamsArgType) -> xr.DataArray:
        complete_param_set = self.fill_with_defaults(params)
        return self[res_name].loc[complete_param_set] >= 0

    def compute(
        self, res_name: str, params: ParamsSingleValue, **add_kwargs: dict[str, Any]
    ) -> Any:
        """Compute result `res_name` for set of parameters `params`.

        Parameters
        ----------
        res_name : str
            Name of the result to compute.
        params : ParamsSingleValue
            Dictionary of parameters for which to perform the computation.
        **add_kwargs : dict[str, Any]
            Additional keyword arguments to pass to `res_name`'s computing function,
            like external data for instance.

        Returns
        -------
        result value : Any
        """
        complete_param_set = self.fill_with_defaults(params)
        self._verbose_print(
            f"Computing {res_name} for the following parameter values:\n{complete_param_set}"
        )
        # Add other results to add_kwargs if necessary.
        argspec = inspect.getfullargspec(self[res_name].compute_fun)
        possible_kwds = set(argspec.args + argspec.kwonlyargs)
        res_names = set(self.param_space.data_vars.keys())
        other_res_deps = {
            rn: self.get(rn, complete_param_set, **add_kwargs)
            for rn in possible_kwds.intersection(res_names)
        }
        add_kwargs = {**add_kwargs, **other_res_deps}
        result = self[res_name].tracking_compute_fun(**complete_param_set, **add_kwargs)

        self._post_compute(res_name, complete_param_set)
        return result

    def _post_compute(
        self, res_name: str, complete_param_set: ParamsSingleValue
    ) -> None:
        self.add_param_values(complete_param_set)
        res_idx = len(self[res_name].attrs["computed_values"]) - 1
        self[res_name].loc[complete_param_set] = res_idx

    def get(
        self, res_name: str, params: ParamsSingleValue, **add_kwargs: dict[str, Any]
    ) -> Any:
        """Get the value of result `res_name` for set of parameters `params`.

        If it has not been computed yet, it will be computed before returning the value.

        Parameters
        ----------
        res_name : str
            Name of the result to get.
        params : ParamsSingleValue
            Dictionary of parameters for which to get the result.
        **add_kwargs : dict[str, Any]
            Additional keyword arguments to pass to `res_name`'s computing function,
            like external data for instance.

        Returns
        -------
        result value : Any
        """
        complete_param_set = self.fill_with_defaults(params)
        res_idx = self[res_name].loc[complete_param_set].values
        if res_idx >= 0:
            return self[res_name].attrs["computed_values"][res_idx]

        return self.compute(res_name, complete_param_set, **add_kwargs)

    def get_time(self, res_name: str, params: ParamsSingleValue) -> float:
        """Get the computing time of result `res_name` for parameters `params`.

        It must have been computed beforehand.

        Parameters
        ----------
        res_name : str
            Name of the result for which to get the computing time.
        params : ParamsSingleValue
            Dictionary of parameters for which to get the result.

        Returns
        -------
        float
            Computing time, in seconds

        Raises
        ------
        ValueError
            If the result has never been computed.
        """
        complete_param_set = self.fill_with_defaults(params)
        res_idx = self[res_name].loc[complete_param_set].values
        if res_idx >= 0:
            time: float = self[res_name].attrs["compute_times"][res_idx]
            return time
        else:
            raise ValueError(f"{res_name} was not computed for {complete_param_set}.")

    def set(
        self,
        res_name: str,
        value: Any,
        params: ParamsSingleValue,
        compute_time: float = np.nan,
    ) -> None:
        """Set value `value` for result `res_name` for set of parameters `params`.

        Parameters
        ----------
        res_name : str
            Name of the result to set.
        value : Any
            Value of the result to set.
        params : ParamsSingleValue
            Dictionary of parameters for which to set the result.
        compute_time : float, optional
            Time taken to compute this value, left as `numpy.nan` if unspecified.
        """
        complete_param_set = self.fill_with_defaults(params)
        self[res_name].attrs["computed_values"].append(value)
        self[res_name].attrs["compute_times"].append(compute_time)
        self._post_compute(res_name, complete_param_set)

    def save(
        self,
        res_name: str,
        params: ParamsSingleValue,
        save_path_fmt: Path | str | None = None,
        **add_kwargs: dict[str, Any],
    ) -> Any:
        """Save the value of result `res_name` for set of parameters `params`.

        If it has not been computed yet, it will be computed before saving the value.

        Parameters
        ----------
        res_name : str
            Name of the result to save.
        params : ParamsSingleValue
            Dictionary of parameters for which to save the result.
        save_path_fmt : Path | str | None, optional
            Format for the path where to save the result. If not set, will be the
            result's default `save_path_fmt`, or, if not set, the global
            :attr:`ResultSet.save_path_fmt`.
        **add_kwargs : dict[str, Any]
            Additional keyword arguments to pass to `res_name`'s computing function,
            like external data for instance.

        Returns
        -------
        result value : Any
        """
        save_fun = self[res_name].attrs["save_fun"]
        save_path = self.get_save_path(res_name, params, save_path_fmt=save_path_fmt)
        res_value = self.get(res_name, params, **add_kwargs)
        self._verbose_print(f"Saving {res_name} at {save_path}.")
        save_fun(res_value, save_path)
        return res_value

    def get_save_path(
        self,
        res_name: str,
        params: ParamsSingleValue,
        save_path_fmt: Path | str | None = None,
    ) -> Path:
        if save_path_fmt is None:
            save_path_fmt = (
                self[res_name].attrs.get("save_path_fmt") or self.save_path_fmt
            )

        save_path_fmt = str(save_path_fmt)
        complete_param_set = self.fill_with_defaults(params)
        save_path = Path(save_path_fmt.format(res_name=res_name, **complete_param_set))
        save_suffix = self[res_name].attrs["save_suffix"]
        return save_path.with_suffix(save_suffix)

    def load(
        self,
        res_name: str,
        params: ParamsSingleValue | None = None,
        save_path: Path | str | None = None,
        save_path_fmt: Path | str | None = None,
    ) -> Any:
        if save_path is None:
            if params is None:
                raise ValueError("Specify either `save_path` or `params`.")
            save_path = self.get_save_path(res_name, params, save_path_fmt)
        load_fun = self[res_name].attrs["load_fun"]
        res = load_fun(save_path)
        return res

    def get_nth_last_result(self, res_name: str, n: int = 1) -> Any:
        """Get the nth last computed value of result `res_name`.

        Parameters
        ----------
        res_name : str
            Name of the result for which to get the value.
        n : int
            Negative index for which to get the value, meaning the method will return
            list_of_values[-n]. By default the value computed last will be returned.

        Returns
        -------
        Any
            nth last value of result `res_name`
        """
        return self[res_name].attrs["computed_values"][-n]

    def get_nth_last_time(self, res_name: str, n: int = 1) -> float:
        """Get the computing time for the nth last computation of result `res_name`.

        Parameters
        ----------
        res_name : str
            Name of the result for which to get the computing time.
        n : int
            Negative index for which to get the computing time, meaning the method will
            return list_of_times[-n]. By default the time of the last computation
            will be returned.

        Returns
        -------
        float
            nth last computation time of result `res_name`
        """
        time: float = self[res_name].attrs["compute_times"][-n]
        return time

    def get_nth_last_params(
        self, res_name: str, n: int = 1
    ) -> dict[Hashable, Hashable]:
        """Get the dictionary of parameters for the nth last computation of `res_name`.

        Parameters
        ----------
        res_name : str
            Name of the result for which to get the parameters.
        n : int
            Negative index for which to get the parameters, meaning the method will
            return list_of_parameters[-n]. By default the parameters of the last
            computation will be returned.

        Returns
        -------
        dict[Hashable, Hashable]
            Dictionary with the parameters of the nth last computation of result
            `res_name`.
        """
        match_idx = len(self[res_name].attrs["computed_values"]) - n if n > 0 else n
        params_idc = np.nonzero(self[res_name].data == match_idx)
        params = {}
        for i, d in enumerate(self.coords.keys()):
            params[d] = self.coords[d].data[params_idc[i]][0]
        return params

    def get_all_computed_values(self, res_name: str) -> dict[str, list[Any]]:
        """Get a dictionary giving all computed values and their parameters.

        Parameters
        ----------
        res_name : str
            Name of the result for which to get this dictionary.

        Returns
        -------
        dict[str, list[Any]]
            Dictionary, one key of which is the one passed with `res_name`, and the
            corresponding value is the complete list of computed values for this result.
            The other keys correspond to the parameter names, and the associated values
            are the list of their values which were used to compute the corresponding
            result.
        """
        flat_populated_space = self[res_name].stack(pset=list(self.coords))
        flat_populated_space = flat_populated_space[flat_populated_space >= 0]
        output_dict = {res_name: flat_populated_space.attrs["computed_values"]}
        for p in self.parameters:
            output_dict[p.name] = flat_populated_space.coords[p.name].values.tolist()
        return output_dict

    def get_timing_stats(
        self, res_names: str | list[str] | None = None
    ) -> pd.DataFrame:
        """Get the summary statistics of the distirbutions of computing times for results.

        Parameters
        ----------
        res_names : str | list[str] | None, optional
            Name of the result or list of names of the results for which to get the
            timing statistics. By default, the statistics will be returned for all
            results.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the statistics on computing times returned by
            :meth:`pd.Series.describe`.
        """
        if res_names is None:
            res_names = list(self.param_space.data_vars.keys())
        elif isinstance(res_names, str):
            res_names = [res_names]

        stat_df = pd.concat(
            [
                pd.Series(self[r].attrs["compute_times"], name=r).describe()
                for r in res_names
            ],
            axis=1,
        )
        return stat_df.T

    def rank_longest_to_compute(
        self, res_name: str, params_as_idx: bool = False
    ) -> pd.DataFrame:
        """Return the ranking of parameters from longest to compute to shortest.

        Parameters
        ----------
        res_name : str
            Name of the result for which to get the ranking.
        params_as_idx : bool
            Whether to have the parameters as part of a :class:`pd.MultiIndex` in the
            output DataFrame. By default, they will be as columns.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the ranking of computing times for result `res_name`,
            as well as the parameters associated to these computations.
        """
        times: list[float] = self[res_name].attrs["compute_times"]
        times_ordering = np.argsort(times)
        ranking_list = []
        for o in times_ordering:
            ranking_list.append(self.get_nth_last_params(res_name, n=-o))
        params_df = pd.DataFrame(ranking_list)
        if params_as_idx:
            ranking_df = pd.DataFrame.from_records(
                data={"compute_times": times},
                index=pd.MultiIndex.from_frame(params_df),
            )
        else:
            ranking_df = params_df.assign(compute_times=times)
        return ranking_df

    def add_param_values(self, params_values: ParamsArgType) -> None:
        """Add new values to existing parameters.

        Parameters
        ----------
        params_values : ParamsArgType
            Dictionary whose keys are the parameter names, and the values either a
            single new value for the parameter, or a sequence of them. Values that are
            already present will be ignored.
        """
        reindex_dict = {}
        for p, v in params_values.items():
            collec = [v] if isinstance(v, Hashable) else v
            reindex_dict[p] = self.param_space.get_index(p).union(pd.Index(collec))

        self.param_space = self.param_space.reindex(reindex_dict, fill_value=-1).copy()

    def add_params(self, params: ParamsType) -> None:
        """Add new parameters to the set.

        Parameter dimensions are always added in such a way that param_space.dims is
        ordered.

        Parameters
        ----------
        params : ParamsType
            Parameters to add.
        """
        if isinstance(params, ParameterSet):
            params_set = params
        else:
            params_set = ParameterSet(params)
        # Type ignore below because don't know how to tell mypy we've locked coords so
        # that parameter labels as returned here below can only be strings.
        curr_dims = list(self.coords)
        add_dims = np.asarray([p.name for p in params_set])
        add_dims_sorting = np.argsort(add_dims)
        # This works because curr_dims is assumed always sorted
        axis_of_sorted_added_dims = np.searchsorted(
            curr_dims, add_dims[add_dims_sorting]  # type: ignore[arg-type]
        ) + np.arange(add_dims.size)
        axis = axis_of_sorted_added_dims[add_dims_sorting].tolist()
        params_dict = params_set.to_dict()
        self.param_space = self.param_space.expand_dims(params_dict, axis=axis)

    def add_results(
        self,
        results_metadata: ResultSetMetadata | ResultSetMetadataInput,
    ) -> None:
        """Add new results to the set.

        Parameters
        ----------
        results_metadata : ResultSetMetadata | ResultSetMetadataInput
            :class:`ResultSetMetadata` instance or dictionary describing the results to
            add. See :meth:`ResultSetMetadata.from_dict` for more information on the
            dictionary format.
        """
        if not isinstance(results_metadata, ResultSetMetadata):
            results_metadata = ResultSetMetadata(results_metadata)

        dims = list(self.coords)
        data = -np.ones([len(p.values) for p in self.parameters], dtype="int")
        add_data_vars = {
            r.name: self._make_res_var(r, dims, data) for r in results_metadata
        }
        self.param_space = self.param_space.assign(variables=add_data_vars)

    def _make_res_var(
        self, res: ResultMetadata, dims: Sequence[Hashable], data: npt.NDArray[np.int_]
    ) -> xr.Variable:
        return xr.Variable(
            dims,
            data,
            attrs={
                "tracking_compute_fun": _tracking(self, res.name)(res.compute_fun),
                "computed_values": [],
                "compute_times": [],
                **asdict(res),
            },
        )  # type: ignore[no-untyped-call]

    @property
    def populated_mask(self) -> xr.Dataset:
        return self.param_space >= 0

    @property
    def populated_space(self) -> xr.Dataset:
        return self.param_space.where(self.populated_mask, drop=True)

    def get_subspace_res(
        self,
        subspace_params: ParamsArgType,
        keep_others_default_only: bool = False,
    ) -> ResultSet:
        """Return the subset of the result set for the given parameter values.

        Parameters
        ----------
        subspace_params : ParamsArgType
            Dictionary giving some parameters' subset of values to keep.
        keep_others_default_only : bool
            If False, the default, for parameters not specified in `subspace_params`,
            keep only the coordinate of their defaults values. If True, keep all their
            values.

        Returns
        -------
        ResultSet
        """
        if keep_others_default_only:
            complete_param_set = self.fill_with_defaults(subspace_params)
        else:
            complete_param_set = {**self.params_values, **subspace_params}
        param_subspace = self.param_space.loc[complete_param_set]
        subspace_res = ResultSet(
            results_metadata=self.results, params=complete_param_set
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
            subspace_res[res_name].data = param_data
            idc_res = np.unique(param_subspace[res_name])
            subspace_res[res_name].attrs["computed_values"] = [
                self[res_name].attrs["computed_values"][i] for i in idc_res[1:]
            ]
            subspace_res[res_name].attrs["compute_times"] = [
                self[res_name].attrs["compute_times"][i] for i in idc_res[1:]
            ]
        return subspace_res
