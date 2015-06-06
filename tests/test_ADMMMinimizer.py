import numpy as np
from src.objects.minimizer import Minimizer
from tests.test_Minimizer import generate_data
from src.objects.admm_minimizer import ADMMMinimizer
from src.functions.matrices import left_multiply_Z
import timeit

def setUpADMMMinimizer(m, n, ratio_of_max=0.1):
    cp, weights, initial_guess, y, phi_indices = generate_data(m, n)
    admm_lambda_max = np.max(np.absolute(left_multiply_Z(y, phi_indices, n, transpose=True)))
    return ADMMMinimizer(cp, weights, ratio_of_max * admm_lambda_max, y, phi_indices)

def true_solution(admm_minimizer):
    minimizer = Minimizer(admm_minimizer.coding_params, admm_minimizer.weights, admm_minimizer.initial_guess,
                          1.0 / (2.0 * admm_minimizer.admm_lambda), admm_minimizer.y, admm_minimizer.phi_indices)
    return minimizer.solve()

#### Generic Tests
def rank_m_inverse(admm_minimizer):
    assert np.allclose(np.eye(admm_minimizer.m),
                       np.dot(admm_minimizer.get('rank_m_matrix'), admm_minimizer.rank_m_inverse))

def update_matrix(admm_minimizer):
    assert np.allclose(np.eye(admm_minimizer.n),
                       np.dot(admm_minimizer.get('update_matrix'), admm_minimizer.get('update_matrix_inv')))

def left_multiply_update_matrix_inv(admm_minimizer):
    x = np.random.randn(admm_minimizer.n)
    assert np.allclose(admm_minimizer.left_multiply_update_inverse(x),
                       np.dot(admm_minimizer.get('update_matrix_inv'), x))

def solve(admm_minimizer):
    start = timeit.default_timer()
    x_star, p_star = admm_minimizer.solve()
    end = timeit.default_timer()
    print "Mine took " + str(end - start) + " for " + str(admm_minimizer.iterations) + " iterations"

    start = timeit.default_timer()
    x_star_cvx, p_star_cvx = true_solution(admm_minimizer)
    end = timeit.default_timer()
    print "CVX took " + str(end - start)
    print "CVX pstar is " + str(p_star_cvx) + " while mine is " + str(p_star)
    # assert np.allclose(x_star, x_star_cvx, atol=10**-4.0)
    assert np.isclose(p_star, p_star_cvx, atol=10**-2.0)

def run_quadratic(fcn, m, n):
    admm_minimizer = setUpADMMMinimizer(m, n, ratio_of_max=0.1)
    fcn(admm_minimizer)

def run_through_many_sizes(fcn, m_list, c_list):
    for m in m_list:
        for c in c_list:
            n = m * c
            run_quadratic(fcn, m, n)

class TestADMMMinimizer:
    def setUp(self):
        self.m_list = np.power(2, np.linspace(1, 6, 6))
        self.compression_list = np.power(2, np.linspace(1, 3, 3))

    def test_rank_m_invers(self):
        run_through_many_sizes(rank_m_inverse, self.m_list, self.compression_list)

    def test_update_matrix(self):
        run_through_many_sizes(update_matrix, self.m_list, self.compression_list)

    def test_left_multiply_update_matrix_inv(self):
        run_through_many_sizes(left_multiply_update_matrix_inv, self.m_list, self.compression_list)

    def test_solve(self):
        run_through_many_sizes(solve, self.m_list, self.compression_list)

