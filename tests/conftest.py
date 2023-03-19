import time

import pytest
from respace import (
    Parameter,
    ParameterSet,
    ResultMetadata,
    ResultSet,
    ResultSetMetadata,
)


def add_one(p):
    return p + 1


def add_r1(p1, p2, r1):
    return p1 + p2 + r1


def sleep(p):
    return time.sleep(p)


def current_time(p):
    return time.time()


@pytest.fixture()
def simple_parameter():
    return Parameter("p", 1)


@pytest.fixture()
def simple_parameter_set(simple_parameter):
    return ParameterSet(simple_parameter)


@pytest.fixture()
def simple_result_metadata():
    return ResultMetadata("r", add_one)


@pytest.fixture()
def simple_result_set_metadata(simple_result_metadata):
    return ResultSetMetadata(simple_result_metadata)


@pytest.fixture()
def simple_result_set(simple_result_set_metadata, simple_parameter_set):
    return ResultSet(simple_result_set_metadata, simple_parameter_set)


@pytest.fixture()
def sleeping_result_set():
    result_set_metadata = ResultSetMetadata(ResultMetadata("r", sleep))
    return ResultSet(result_set_metadata, ParameterSet(Parameter("p", 0.01)))


@pytest.fixture()
def current_time_result_set(simple_parameter_set):
    result_set_metadata = ResultSetMetadata(ResultMetadata("r", current_time))
    return ResultSet(result_set_metadata, simple_parameter_set)


@pytest.fixture()
def generic_parameter_set(simple_parameter):
    return ParameterSet(
        [simple_parameter, Parameter("p1", 2, [3, 4]), Parameter("p2", -1, [0, -1])]
    )


@pytest.fixture()
def generic_result_set_metadata(simple_result_metadata):
    return ResultSetMetadata(
        [
            simple_result_metadata,
            ResultMetadata("r1", add_one),
            ResultMetadata("r2", add_r1),
        ]
    )


@pytest.fixture()
def generic_result_set(generic_result_set_metadata, generic_parameter_set):
    return ResultSet(generic_result_set_metadata, generic_parameter_set)
