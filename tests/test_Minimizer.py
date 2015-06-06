import numpy as np
from src.objects.minimizer import Minimizer
from src.objects.audiofile import CodingParams
from src.functions.matrices import create_Z, RandomSubset
from params import setCodingParams, setDecodingParams

def generate_data(m, n):
    np.random.seed()
    phi_indices = RandomSubset(n, m)
    y = np.random.randn(m)
    weights = 10 * np.random.rand(n)

    cp = CodingParams()
    cp = setDecodingParams(cp)
    cp = setCodingParams(cp)
    cp.full_block_length = n
    cp.compressed_block_length = m

    Z = create_Z(phi_indices, n)
    initial_guess = np.linalg.lstsq(Z, y)[0]

    return cp, weights, initial_guess, y, phi_indices

def setUpMinimizer(m, n, lambd=10.0):
    cp, weights, initial_guess, y, phi_indices = generate_data(m, n)
    return Minimizer(cp, weights, initial_guess, lambd=lambd, y=y, phi_indices=phi_indices)

def is_deterministic(minimizer):
    xstar1, pstar1 = minimizer.solve()
    xstar2, pstar2 = minimizer.solve()
    assert np.all(xstar1 == xstar2)
    assert pstar1 == pstar2

def run_affine_and_quadratic(fcn, m, n):
    # Affine
    minimizer = setUpMinimizer(m, n, lambd=0.0)
    fcn(minimizer)

    # Quadratic
    minimizer = setUpMinimizer(m, n, lambd=10.0)
    fcn(minimizer)

def run_through_many_sizes(fcn, m_list, c_list):
    for m in m_list:
        for c in c_list:
            n = m * c
            run_affine_and_quadratic(fcn, m, n)

class TestMinimizer:
    def setUp(self):
        self.m_list = np.power(2, np.linspace(1, 6, 6))
        self.compression_list = np.power(2, np.linspace(1, 3, 3))

    def test_is_deterministic(self):
        run_through_many_sizes(is_deterministic, self.m_list, self.compression_list)

