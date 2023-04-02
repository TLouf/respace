from respace import ResultSet, tracking


def test_tracking_new_result(simple_parameter, simple_result_metadata):
    res_name = simple_result_metadata.name
    res_fun = simple_result_metadata.compute_fun
    rs = ResultSet()
    tracking_fun = tracking(rs, res_name)(res_fun)
    assert rs.results.results == [simple_result_metadata]

    rs.add_params({simple_parameter.name: 1})
    r = tracking_fun(p=2)
    assert rs[res_name].attrs["computed_values"] == [res_fun(2)]
    assert r == res_fun(2)

    r = tracking_fun(3)
    assert rs[res_name].attrs["computed_values"] == [res_fun(2), res_fun(3)]
    assert r == res_fun(3)


def test_tracking_existing_result(simple_result_metadata, simple_result_set):
    res_name = simple_result_metadata.name
    res_fun = simple_result_metadata.compute_fun
    rs = simple_result_set
    tracking_fun = tracking(rs, res_name)(res_fun)
    assert rs.results.results == [simple_result_metadata]

    r = tracking_fun(p=2)
    assert rs[res_name].attrs["computed_values"] == [res_fun(2)]
    assert r == res_fun(2)

    def new_fun(p):
        return p + 2

    tracking_fun = tracking(rs, res_name)(new_fun)
    simple_result_metadata.compute_fun = new_fun
    assert rs.results.results == [simple_result_metadata]
    r = tracking_fun(p=2)
    assert rs[res_name].attrs["computed_values"] == [new_fun(2)]
    assert r == new_fun(2)
