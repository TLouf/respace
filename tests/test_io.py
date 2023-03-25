from respace import load_pickle, save_pickle


def test_pickle(tmp_path):
    a = 1
    save_path = tmp_path / "a.pickle"
    save_pickle(a, save_path)
    assert load_pickle(save_path) == a
