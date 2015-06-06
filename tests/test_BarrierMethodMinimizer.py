import numpy as np
import cvxpy as cvx
from src.objects.minimizer import Minimizer
from tests.test_Minimizer import generate_data
from src.objects.barrier_method_minimizer import BarrierMethodMinimizer
from src.objects.newton_iteration_data import NewtonIterationData
import timeit

def setUpBMMinimizer(m, n, lambd=10.0, initial_t=1.0):
    cp, weights, initial_guess, y, phi_indices = generate_data(m, n)
    return BarrierMethodMinimizer(cp, weights, initial_guess, initial_t, lambd=lambd, y=y, phi_indices=phi_indices)

def true_solution(bm_minimizer):
    minimizer = Minimizer(bm_minimizer.coding_params, bm_minimizer.weights, bm_minimizer.initial_guess,
                          bm_minimizer.lambd, bm_minimizer.y, bm_minimizer.phi_indices)
    return minimizer.solve()

def z_objective_cvxpy(z, bm_minimizer):
    return cvx.sum_entries(z[bm_minimizer.n:]) + (bm_minimizer.lambd * cvx.norm(bm_minimizer.y - bm_minimizer.get('Z') * z[0:bm_minimizer.n])**2.0 if bm_minimizer.quadratic else 0)

def lin_ineq(z, bm_minimizer):
    return bm_minimizer.get('A') * z

def barrier_fcn_cvxpy(z, bm_minimizer):
    return -1.0 * cvx.sum_entries(cvx.log(-1.0 * lin_ineq(z, bm_minimizer)))

def newton_obj_cvxpy(z, bm_minimizer):
    return bm_minimizer.t * z_objective_cvxpy(z, bm_minimizer) + barrier_fcn_cvxpy(z, bm_minimizer)

def minimize_centering_step_with_cvxpy(bm_minimizer):
    zhat = cvx.Variable(2 * bm_minimizer.n)
    objective = cvx.Minimize(newton_obj_cvxpy(zhat, bm_minimizer))
    problem = cvx.Problem(objective)
    pstar = problem.solve()
    return np.array(zhat.value.transpose())[0], pstar

#### Generic Tests
def always_feasible(bm_minimizer):
    while not bm_minimizer.optimal:
        assert bm_minimizer.iter_data.feasible
        bm_minimizer.iterate()

def hess_psd(bm_minimizer):
    while not bm_minimizer.optimal:
        assert np.all(np.linalg.eigvals(bm_minimizer.iter_data.calc_hess_newton_obj()) > 0)
        bm_minimizer.iterate()

def descent(bm_minimizer):
    while True:
        prev_obj = bm_minimizer.iter_data.get('newton_obj')
        bm_minimizer.iterate()
        if bm_minimizer.optimal_nt:
            break
        assert bm_minimizer.iter_data.get('newton_obj') < prev_obj

def solve(bm_minimizer):
    start = timeit.default_timer()
    x_star, p_star = bm_minimizer.solve()
    end = timeit.default_timer()
    print "\n" + ("QUADRATIC: " if bm_minimizer.quadratic else "AFFINE: ") + "for n = " + str(bm_minimizer.n) + ", m = " + str(bm_minimizer.m)
    print "Mine took " + str(end - start)

    start = timeit.default_timer()
    x_star_cvx, p_star_cvx = true_solution(bm_minimizer)
    end = timeit.default_timer()
    print "CVX took " + str(end - start)
    print "max diff is " + str(np.max(np.absolute(x_star - x_star_cvx)))
    print "CVX pstar is " + str(p_star_cvx) + " while mine is " + str(p_star)
    assert np.allclose(x_star, x_star_cvx, atol=10**-2.0)
    assert np.isclose(p_star, p_star_cvx, atol=10**-2.0)

def run_affine_and_quadratic(fcn, m, n):
    # Affine
    bm_minimizer = setUpBMMinimizer(m, n, lambd=0.0)
    fcn(bm_minimizer)

    # Quadratic
    bm_minimizer = setUpBMMinimizer(m, n, lambd=10.0)
    fcn(bm_minimizer)

def run_through_many_sizes(fcn, m_list, c_list):
    for m in m_list:
        for c in c_list:
            n = m * c
            run_affine_and_quadratic(fcn, m, n)

class TestBarrierMethodMinimizer:
    def setUp(self):
        self.m_list = np.power(2, np.linspace(1, 6, 6))
        self.compression_list = np.power(2, np.linspace(1, 3, 3))

    def test_always_feasible(self):
        run_through_many_sizes(always_feasible, self.m_list, self.compression_list)

    # def test_hess_psd(self): # Reallyyyyyy slow
    #     run_through_many_sizes(hess_psd, self.m_list, self.compression_list)

    def test_descent(self):
        run_through_many_sizes(descent, self.m_list, self.compression_list)

    def test_l1_equivalence(self):
        for m in self.m_list:
            for c in self.compression_list:
                n = int(m * c)
                bm_minimizer = setUpBMMinimizer(m, n, lambd=0.0)

                # 1 norm explicitly
                l1_norm_xstar, l1_norm_pstar = true_solution(bm_minimizer)

                # LP
                zhat = cvx.Variable(2 * n)
                constraints = [lin_ineq(zhat, bm_minimizer) < 0]
                objective = cvx.Minimize(z_objective_cvxpy(zhat, bm_minimizer))
                problem = cvx.Problem(objective, constraints)
                lp_pstar = problem.solve()
                lp_xstar = np.array(zhat.value.transpose())[0]
                assert np.allclose(l1_norm_xstar, lp_xstar[0:n])
                assert np.allclose(lp_xstar[0:n], lp_xstar[n:])
                assert np.isclose(lp_pstar, l1_norm_pstar)

    def test_solve(self):
        run_through_many_sizes(solve, self.m_list, self.compression_list)

