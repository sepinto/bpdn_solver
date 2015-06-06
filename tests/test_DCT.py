import numpy as np
from src.functions.dct import *


class TestDCT():
    def setUp(self):
        self.N_list = np.power(2, np.linspace(1, 11, 11))

    def test_DCT_preserves_length(self):
        for N in self.N_list:
            x = np.random.rand(N)
            X = DCT(x)
            assert len(X) == N

    def test_IDCT_preserves_length(self):
        for N in self.N_list:
            x = np.random.rand(N)
            X = IDCT(x)
            assert len(X) == N

    def test_DCT_is_orthogonal(self):
        for N in self.N_list:
            x = np.random.rand(N)
            x_2 = IDCT(DCT(x))
            assert np.allclose(x, x_2)

    def test_DCT_is_linear(self):
        for N in self.N_list:
            x1 = np.random.rand(N)
            x2 = np.random.rand(N)
            a1 = np.random.rand(1)
            a2 = np.random.rand(1)

            X1 = DCT(a1 * x1 + a2 * x2)
            X2 = a1 * DCT(x1) + a2 * DCT(x2)
            assert np.allclose(X1, X2)

            X1 = DCT(a1 * x1 + a2 * x2, isInverse=True)
            X2 = a1 * DCT(x1, isInverse=True) + a2 * DCT(x2, isInverse=True)
            assert np.allclose(X1, X2)

    def test_IDCTmat(self):
        for N in self.N_list:
            X = np.random.rand(N)
            x = IDCT(X)
            x_mat = np.dot(IDCTmat(N), X)
            assert np.allclose(x, x_mat)

    def test_IDCTmat_is_orthogonal(self):
        for N in self.N_list:
            assert np.allclose(np.dot(IDCTmat(N), IDCTmat(N).transpose()), np.eye(N))