import pytest
import xarray as xr
from respace import (
    Parameter,
    ParameterSet,
    ResultMetadata,
    ResultSet,
    ResultSetMetadata,
)
from respace.utils import load_pickle, save_pickle


def one(*args, **kwargs):
    return 1


result_metadata = ResultMetadata("res", one)
result_set_metadata = ResultSetMetadata(result_metadata)
result_metadata_dict = {"res": one}
parameter = Parameter("param", 1)
parameter_set = ParameterSet(parameter)
parameter_dict = {"param": 1}


class TestDataFrameConstructors:
    def setup_method(self):
        self.attrs = {
            "computed_values": [],
            "compute_times": [],
            "name": "res",
            "compute_fun": one,
            "save_fun": save_pickle,
            "load_fun": load_pickle,
            "save_suffix": ".pickle",
            "save_path_fmt": None,
        }
        self.expected_data_array = xr.DataArray(
            -1,
            {parameter.name: [parameter.default]},
            dims=[parameter.name],
            name=result_metadata.name,
            attrs=self.attrs,
        )
        self.expected_dataset = xr.Dataset(
            data_vars={result_metadata.name: self.expected_data_array}
        )

    @pytest.mark.parametrize(
        "results_metadata",
        [result_metadata, [result_metadata], result_set_metadata, result_metadata_dict],
    )
    def test_results_metadata(self, results_metadata):
        rs = ResultSet(results_metadata, parameter)
        # rs[result_metadata.name].attrs.pop('tracking_compute_fun')
        res_name = result_metadata.name
        xr.testing.assert_equal(rs.param_space, self.expected_dataset)
        attrs = rs[res_name].attrs
        expected_attrs = self.attrs.copy()
        # expected_attrs["tracking_compute_fun"] = _tracking(rs, res_name)(expected_attrs["compute_fun"])
        attrs.pop("tracking_compute_fun")
        assert attrs == expected_attrs

    @pytest.mark.parametrize(
        "parameters",
        [parameter, [parameter], parameter_set, parameter_dict],
    )
    def test_parameters(self, parameters):
        rs = ResultSet(result_set_metadata, parameters)
        xr.testing.assert_equal(rs.param_space, self.expected_dataset)
