import time

import pytest


def test_compute_default_value(simple_result_set, simple_parameter_set):
    existing_p = simple_result_set.coords["p"].values.tolist()
    p = existing_p[0]
    value = simple_result_set.compute("r", {})
    assert value == p + 1
    assert simple_result_set["r"].attrs["computed_values"] == [value]
    assert simple_result_set["r"].loc[{"p": p}].values == 0
    assert simple_result_set.parameters == simple_parameter_set


def test_compute_new_value(simple_result_set, simple_parameter_set):
    existing_p = simple_result_set.coords["p"].values.tolist()
    new_p = existing_p[-1] + 1
    value = simple_result_set.compute("r", {"p": new_p})
    assert value == new_p + 1
    assert simple_result_set["r"].attrs["computed_values"] == [value]
    assert simple_result_set["r"].loc[{"p": new_p}].values == 0
    expected_pset = simple_parameter_set.copy()
    expected_pset[0].values.append(2)
    assert simple_result_set.parameters == expected_pset


def test_compute_raises_for_unknown_parameter(simple_result_set):
    msg = "One of the passed parameters is not present in the set."
    with pytest.raises(KeyError, match=msg):
        simple_result_set.compute("r", {"unknown_p": 1})


def test_get_existing_value(current_time_result_set):
    current_time = current_time_result_set.compute("r", {})
    new_current_time = current_time_result_set.get("r", {})
    assert current_time == new_current_time


def test_get_new_value(simple_result_set, simple_parameter_set):
    existing_p = simple_result_set.coords["p"].values.tolist()
    p = existing_p[0]
    value = simple_result_set.get("r", {})
    assert value == p + 1
    assert simple_result_set["r"].attrs["computed_values"] == [value]
    assert simple_result_set["r"].loc[{"p": p}].values == 0
    assert simple_result_set.parameters == simple_parameter_set


def test_get_for_new_parameter(simple_result_set, simple_parameter_set):
    existing_p = simple_result_set.coords["p"].values.tolist()
    new_p = existing_p[-1] + 1
    value = simple_result_set.get("r", {"p": new_p})
    assert value == new_p + 1
    assert simple_result_set["r"].attrs["computed_values"] == [value]
    assert simple_result_set["r"].loc[{"p": new_p}].values == 0
    expected_pset = simple_parameter_set.copy()
    expected_pset[0].values.append(2)
    assert simple_result_set.parameters == expected_pset


def test_timing(sleeping_result_set):
    p_dict = {"p": 0.01}
    t1 = time.time()
    sleeping_result_set["r"].tracking_compute_fun(**p_dict)
    t2 = time.time()
    sleeping_result_set._post_compute("r", p_dict)
    tracked_time = sleeping_result_set.get_time("r", p_dict)
    # The following LHS is necessarily positive.
    assert ((t2 - t1) - tracked_time) / tracked_time < 0.1


def test_get_time(simple_result_set):
    simple_result_set.compute("r", {})
    expected_time = simple_result_set["r"].attrs["compute_times"][0]
    assert simple_result_set.get_time("r", {}) == expected_time


def test_get_time_raises_for_not_computed(simple_result_set):
    with pytest.raises(ValueError, match="r was not computed for"):
        simple_result_set.get_time("r", {})


def test_get_all_computed_values(generic_result_set):
    rs = generic_result_set
    params_for_compute = [{"p1": 3}, {"p": 2, "p2": 0}, {"p": 2, "p1": 3}]
    for params in params_for_compute:
        rs.compute("r", params)
    values = rs.get_all_computed_values("r")
    expected = {"r": rs["r"].attrs["computed_values"]}
    for p in generic_result_set.parameters:
        expected[p.name] = [
            p_dict.get(p.name, p.default) for p_dict in params_for_compute
        ]
    assert values == expected


def test_set_new_value(simple_result_set, simple_parameter_set):
    existing_p = simple_result_set.coords["p"].values.tolist()
    new_p = existing_p[-1] + 1
    value = new_p + 1
    simple_result_set.set("r", value, {"p": new_p}, compute_time=0)
    assert value == new_p + 1
    assert simple_result_set["r"].attrs["computed_values"] == [value]
    assert simple_result_set["r"].attrs["compute_times"] == [0]
    assert simple_result_set["r"].loc[{"p": new_p}].values == 0
    expected_pset = simple_parameter_set.copy()
    expected_pset[0].values.append(2)
    assert simple_result_set.parameters == expected_pset
