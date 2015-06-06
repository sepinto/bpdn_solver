import numpy as np
from src.functions.mdct import *

__author__ = 'spinto'

class TestMDCT:
    def setUp(self):
        self.N_list = np.power(2, np.linspace(1, 11, 11))

    def test_mdct_halves_length(self):
        for N in self.N_list:
            x = np.random.rand(N)
            X = MDCT(x, N/2, N/2, isInverse=False)
            assert len(X) == N/2

    def test_imdct_doubles_length(self):
        for N in self.N_list:
            x = np.random.rand(N)
            X = MDCT(x, N, N, isInverse=True)
            assert len(X) == 2 * N

    def test_linearity(self):
        for N in self.N_list:
            x1 = np.random.rand(N)
            x2 = np.random.rand(N)
            a1 = np.random.rand(1)
            a2 = np.random.rand(1)

            X1 = MDCT(a1 * x1 + a2 * x2, N/2, N/2)
            X2 = a1 * MDCT(x1, N/2, N/2) + a2 * MDCT(x2, N/2, N/2)
            assert np.allclose(X1, X2)

            X1 = MDCT(a1 * x1 + a2 * x2, N, N, isInverse=True)
            X2 = a1 * MDCT(x1, N, N, isInverse=True) + a2 * MDCT(x2, N, N, isInverse=True)
            assert np.allclose(X1, X2)

    def test_MDCTmat(self):
        for N in self.N_list:
            X = np.random.rand(N)

            x = IMDCT(X, N, N)
            x_mat = np.dot(IMDCTmat(2 * N), X)
            assert np.allclose(x, x_mat)

            y = MDCT(X, N/2, N/2)
            y_mat = np.dot(MDCTmat(N/2, N/2, False), X)
            assert np.allclose(y, y_mat)