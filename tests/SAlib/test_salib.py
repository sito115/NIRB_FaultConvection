import plotly.graph_objects as go
import numpy as np
import numpy.testing as npt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))
# import Salib from module
from SALib.sample import morris as morris_sample
from SALib.analyze import morris
from SALib.test_functions import Ishigami
from SALib.util import read_param_file

from SALib.analyze.morris import (
    analyze,
    _compute_mu_star_confidence,
    _compute_elementary_effects,
    _reorganize_output_matrix,
    _compute_grouped_metric,
    _compute_grouped_sigma,
    _check_if_array_of_floats,
)


from pytest import raises, mark
from numpy.testing import assert_equal, assert_allclose
import numpy as np
from scipy.stats import norm

def test_compute_mu_star_confidence():
    """
    Tests that compute mu_star_confidence is computed correctly
    """

    ee = np.array([[2.52, 2.01, 2.30, 0.66, 0.93, 1.3]], dtype=float)
    num_resamples = 1000
    conf_level = 0.95
    num_vars = 1

    actual = _compute_mu_star_confidence(ee, num_vars, num_resamples, conf_level)
    expected = 0.5
    assert_allclose(actual, expected, atol=1e-01)
    
    
def test_analysis_of_morris_results():
    """
    Tests a one-dimensional vector of results

    Taken from the solution to Exercise 4 (p.138) in Saltelli (2008).
    """
    model_input = np.array(
        [
            [0, 1.0 / 3],
            [0, 1],
            [2.0 / 3, 1],
            [0, 1.0 / 3],
            [2.0 / 3, 1.0 / 3],
            [2.0 / 3, 1],
            [2.0 / 3, 0],
            [2.0 / 3, 2.0 / 3],
            [0, 2.0 / 3],
            [1.0 / 3, 1],
            [1, 1],
            [1, 1.0 / 3],
            [1.0 / 3, 1],
            [1.0 / 3, 1.0 / 3],
            [1, 1.0 / 3],
            [1.0 / 3, 2.0 / 3],
            [1.0 / 3, 0],
            [1, 0],
        ],
        dtype=float,
    )

    model_output = np.array(
        [
            0.97,
            0.71,
            2.39,
            0.97,
            2.30,
            2.39,
            1.87,
            2.40,
            0.87,
            2.15,
            1.71,
            1.54,
            2.15,
            2.17,
            1.54,
            2.20,
            1.87,
            1.0,
        ],
        dtype=float,
    )

    problem = {
        "num_vars": 2,
        "names": ["Test 1", "Test 2"],
        "groups": None,
        "bounds": [[0.0, 1.0], [0.0, 1.0]],
    }

    Si, _ = analyze(
        problem,
        model_input,
        model_output,
        num_resamples=1000,
        conf_level=0.95,
        print_to_console=False,
    )

    desired_mu = np.array([0.66, 0.21])
    assert_allclose(
        Si["mu"], desired_mu, rtol=1e-1, err_msg="The values for mu are incorrect"
    )
    desired_mu_star = np.array([1.62, 0.35])
    assert_allclose(
        Si["mu_star"],
        desired_mu_star,
        rtol=1e-2,
        err_msg="The values for mu star are incorrect",
    )
    desired_sigma = np.array([1.79, 0.41])
    assert_allclose(
        Si["sigma"],
        desired_sigma,
        rtol=1e-2,
        err_msg="The values for sigma are incorrect",
    )
    desired_names = ["Test 1", "Test 2"]
    assert_equal(
        Si["names"], desired_names, err_msg="The values for names are incorrect"
    )


def test_additional_info():
    
    N_TRAJECTORIES = 400
    NUM_LEVELS = 8
    tolerance = 1e-10

    
    problem = {
    'names': ['x1', 'x2', 'x3'],
    'num_vars': 3,
    'bounds': [[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]]
    }
    
    param_values = morris_sample.sample(problem,
                      N_TRAJECTORIES,
                      num_levels=NUM_LEVELS,
                      seed = 12300,
                      local_optimization=True)
    Y = Ishigami.evaluate(param_values)
    _ , add_info = morris.analyze(problem, param_values, Y, conf_level=0.95,
                        print_to_console=True, num_levels=NUM_LEVELS, seed=12300)
    result_increased, result_decreased, input_increased, input_decreased = add_info
    
    root = Path(__file__).parent
    result_increased_test = np.load(root / "result_increased.npy")
    result_decreased_test = np.load(root / "result_decreased.npy")
    input_increased_test = np.load( root / "input_increased.npy")
    input_decreased_test = np.load( root / "input_decreased.npy")
    
    
    assert_allclose(result_increased_test, result_increased, atol=tolerance)
    assert_allclose(result_decreased_test, result_decreased, atol=tolerance)
    assert_allclose(input_increased_test, input_increased,   atol=tolerance)
    assert_allclose(input_decreased_test, input_decreased,   atol=tolerance)



if __name__  == "__main__":
    test_additional_info()