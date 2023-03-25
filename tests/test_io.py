from pathlib import Path

import pytest
from respace import load_pickle, save_pickle


def test_pickle(tmp_path):
    a = 1
    save_path = tmp_path / "a.pickle"
    save_pickle(a, save_path)
    assert load_pickle(save_path) == a


def test_save_path(simple_parameter, simple_result_metadata, simple_result_set):
    res_name = simple_result_metadata.name
    p_name = simple_parameter.name
    save_path = simple_result_set.get_save_path(res_name, {p_name: 2})
    assert save_path == Path(f"{res_name}_{p_name}=2.pickle")

    save_path = simple_result_set.get_save_path(res_name, {})
    assert save_path == Path(f"{res_name}_{p_name}={simple_parameter.default}.pickle")

    simple_result_set[res_name].attrs["save_path_fmt"] = "a"
    save_path = simple_result_set.get_save_path(res_name, {p_name: 2})
    assert save_path == Path("a.pickle")

    save_path = simple_result_set.get_save_path(
        res_name, {p_name: 2}, save_path_fmt="b"
    )
    assert save_path == Path("b.pickle")


def test_save_and_load(
    tmp_path, simple_parameter, simple_result_metadata, simple_result_set
):
    res_name = simple_result_metadata.name
    p_name = simple_parameter.name
    params = {p_name: 2}
    simple_result_set.save_path_fmt = str(
        Path(tmp_path / simple_result_set.save_path_fmt)
    )
    saved_value = simple_result_set.save(res_name, params)

    assert simple_result_set.load(res_name, params=params) == saved_value
    save_path = simple_result_set.get_save_path(res_name, params)
    assert simple_result_set.load(res_name, save_path=save_path) == saved_value


def test_load_raises_missing_args(simple_result_set):
    with pytest.raises(ValueError, match="Specify either `save_path` or `params`."):
        simple_result_set.load("r")
