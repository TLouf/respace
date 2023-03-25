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
