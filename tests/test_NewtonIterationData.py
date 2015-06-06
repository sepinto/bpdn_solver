from tests.test_BarrierMethodMinimizer import setUpBMMinimizer
from src.objects.newton_iteration_data import NewtonIterationData
import numpy as np
import timeit

def setUpIterData(m, n):
    bm_minimizer = setUpBMMinimizer(m, n, lambd=10.0)
    iter_data = NewtonIterationData(bm_minimizer.z, bm_minimizer)
    return bm_minimizer, iter_data

def time(fcn, *args):
    t1 = timeit.default_timer()
    result = fcn(*args)
    t2 = timeit.default_timer()
    return t2 - t1, result

class TestNewtonIterationData:
    def setUp(self):
        self.m_list = np.power(2, np.linspace(1, 7, 7))
        self.compression_list = np.power(2, np.linspace(1, 3, 3))

    def test_dictionary_caching(self):
        m = self.m_list[-1]
        c = self.compression_list[-1]
        n = m * c
        bm_minimizer, iter_data = setUpIterData(m, n)

        # Test just once to show the concept works
        assert 'Zx' not in iter_data.vals
        init_time, val1 = time(iter_data.get, 'Zx')
        assert 'Zx' in iter_data.vals
        cache_time, val2 = time(iter_data.get, 'Zx')
        assert cache_time < init_time
        assert np.all(val1 == val2)

    def test_hess_barrier_fcn(self):
        for m in self.m_list:
            for c in self.compression_list:
                n = m * c
                bm_minimizer, iter_data = setUpIterData(m, n)
                # Az is ill formed with massive values if x & s close. Don't care much about accuracy though since
                # we never actually use Az. Az_inv is much better formed with values in more reasonable range.
                assert np.allclose(np.dot(iter_data.get('hess_barrier_fcn'), iter_data.get('hess_barrier_fcn_inv')), np.eye(2 * n),
                                   rtol=np.power(10, -3.0), atol=np.power(10, -3.0))

    def test_left_multiply_Az_inv(self):
        for m in self.m_list:
            for c in self.compression_list:
                n = m * c
                bm_minimizer, iter_data = setUpIterData(m, n)
                x = np.random.randn(2 * n)
                assert np.allclose(iter_data.left_multiply_hess_barrier_fcn_inv(x), np.dot(iter_data.get('hess_barrier_fcn_inv'), x))

    def test_matrix_inversion_lemma(self):
        for m in self.m_list:
            for c in self.compression_list:
                n = m * c
                bm_minimizer, iter_data = setUpIterData(m, n)
                assert np.allclose(iter_data.get('hess_newton_obj_inv'), iter_data.matrix_inversion_lemma())

    def test_left_multiply_hess_barrier_fcn_inv(self):
        for m in self.m_list:
            for c in self.compression_list:
                n = m * c
                bm_minimizer, iter_data = setUpIterData(m, n)
                x = np.random.randn(2 * n)
                mtx = iter_data.get('hess_barrier_fcn_inv')
                assert np.allclose(iter_data.left_multiply_hess_barrier_fcn_inv(x), np.dot(mtx, x),
                                   rtol=np.power(10, -7.0), atol=np.power(10, -7.0))

    def test_left_multiply_hess_newton_obj_inv(self):
        for m in self.m_list:
            for c in self.compression_list:
                n = m * c
                bm_minimizer, iter_data = setUpIterData(m, n)
                x = np.random.randn(2 * n)
                mtx = iter_data.get('hess_newton_obj_inv')
                assert np.allclose(iter_data.left_multiply_hess_newton_obj_inv(x), np.dot(mtx, x),
                                   rtol=np.power(10, -7.0), atol=np.power(10, -7.0))
