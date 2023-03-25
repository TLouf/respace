from respace import ResultSet


def test_parameters(simple_parameter_set, simple_result_set):
    assert simple_result_set.parameters == simple_parameter_set


def test_params_values(simple_parameter_set, simple_result_set):
    expected = {param.name: param.values for param in simple_parameter_set}
    assert simple_result_set.params_values == expected


def test_results(simple_result_set_metadata, simple_result_set):
    assert simple_result_set.results == simple_result_set_metadata


def test_verbose():
    assert ResultSet(verbose=True)._verbose_print == print


def test_save_path_fmt(simple_parameter, simple_result_set):
    p = simple_parameter
    expected = "_".join(["{res_name}", f"{p.name}={{{p.name}}}"])
    assert simple_result_set.save_path_fmt == expected

    simple_result_set.save_path_fmt = ""
    assert simple_result_set.save_path_fmt == ""


def test_set_compute_fun(simple_result_metadata, simple_result_set):
    res_name = simple_result_metadata.name
    res_attrs = simple_result_set[res_name].attrs
    og_tracking_compute_fun = res_attrs["tracking_compute_fun"]

    def expected_compute_fun(p):
        return p + 2

    simple_result_set.set_compute_fun(res_name, expected_compute_fun)
    assert res_attrs["compute_fun"] == expected_compute_fun
    # For now we just check "tracking_compute_fun" has changed.
    assert res_attrs["tracking_compute_fun"] != og_tracking_compute_fun
