import pytest
from respace import Parameter, ParameterSet


@pytest.mark.parametrize(
    "parameter",
    [Parameter("p", 1), Parameter("p", 2, [3, 4]), Parameter("p2", -1, [0, -1])],
)
def test_default_is_first_value(parameter):
    assert parameter.default == parameter.values[0]
    assert parameter.default not in parameter.values[1:]


def test_parameterset_is_sorted():
    params = [Parameter("p2", 1), Parameter("p1", "b")]
    ps = ParameterSet(params)
    assert ps.parameters == params[::-1]

    params = [Parameter("b", 1), Parameter("a", "b")]
    ps.parameters = params
    assert ps.parameters == params[::-1]


def test_parameters_property(generic_result_set, generic_parameter_set):
    assert generic_result_set.parameters == generic_parameter_set


def test_fill_with_defaults(generic_result_set):
    filled_set = generic_result_set.fill_with_defaults({"p1": 5, "absent_parameter": 0})
    assert filled_set == {"absent_parameter": 0, "p": 1, "p1": 5, "p2": -1}


def test_add_values(generic_result_set, generic_parameter_set):
    generic_result_set.add_param_values({"p": 3, "p2": [-1, 1, 2]})
    expected_pset = generic_parameter_set.copy()
    expected_pset[0].values.append(3)
    expected_pset[-1].values.extend([1, 2])
    assert generic_result_set.parameters == expected_pset
    # TODO: add test checking with get_subspace_res that original part of spce is
    # preserved? would have to populate with some computations first.


def test_add_params(simple_result_set, generic_parameter_set):
    simple_result_set.compute("r", {})
    simple_result_set.add_params({"p2": [-1, 0], "p1": [2, 3, 4]})
    assert simple_result_set.parameters == generic_parameter_set

    expected_order = [p.name for p in generic_parameter_set]
    assert list(simple_result_set["r"].dims) == expected_order

    assert simple_result_set.populated_space.sum() == 1
