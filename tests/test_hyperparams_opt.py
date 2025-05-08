import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def test_refine_param_grid(optimize):
    sample_best_params = {
        "n_estimators": 150,
        "max_depth": 10,
        "min_samples_split": 5,
        "some_float_param": 0.1,
        "some_category": "value",
        "none_param": None
    }

    refine = optimize.refine_param_grid(best_params=sample_best_params)

    assert isinstance(refine, dict)

    assert "n_estimators" in refine
    assert isinstance(refine["n_estimators"], list)
    assert 150 * 0.8 <= refine["n_estimators"][0] <= 150 * 1.2

    assert "max_depth" in refine
    assert isinstance(refine["max_depth"], list)
    assert 10 * 0.8 <= refine["max_depth"][0] <= 10 * 1.2

    assert "min_samples_split" in refine
    assert isinstance(refine["min_samples_split"], list)
    assert 5 * 0.8 <= refine["min_samples_split"][0] <= 5 * 1.2

    assert "some_float_param" in refine
    assert isinstance(refine["some_float_param"], tuple)
    assert len(refine["some_float_param"]) == 2
    assert 0.1 * 0.8 <= refine["some_float_param"][0] <= 0.1 * 1.2
    assert 0.1 * 0.8 <= refine["some_float_param"][1] <= 0.1 * 1.2

    assert "some_category" in refine
    assert isinstance(refine["some_category"], list)
    assert refine["some_category"] == ["value"]


def test_with_random_search(optimize):
    rs = optimize.with_random_search()

    assert isinstance(rs, dict)

def test_with_bayesian_search(optimize):
    bs = optimize.with_bayesian_search()
    
    assert isinstance(bs, dict)