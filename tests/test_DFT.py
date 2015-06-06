from unittest import TestCase

import numpy as np

from src.functions.dft import *


class TestDFT(TestCase):
    def setUp(self):
        self.N_list = np.power(2, np.linspace(1, 11, 11))

    def test_DFT_preserves_length(self):
        for N in self.N_list:
            x = np.random.rand(N)
            X = DFT(x)
            assert len(X) == N

    def test_IDFT_preserves_length(self):
        for N in self.N_list:
            x = np.random.rand(N)
            X = IDFT(x)
            assert len(X) == N

    def test_DFT_is_orthogonal(self):
        for N in self.N_list:
            x = np.random.rand(N)
            x_2 = IDFT(DFT(x))
            assert np.allclose(x, x_2)

    def test_DFT_is_linear(self):
        for N in self.N_list:
            x1 = np.random.rand(N)
            x2 = np.random.rand(N)
            a1 = np.random.rand(1)
            a2 = np.random.rand(1)

            X1 = DFT(a1 * x1 + a2 * x2)
            X2 = a1 * DFT(x1) + a2 * DFT(x2)
            assert np.allclose(X1, X2)

            X1 = DFT(a1 * x1 + a2 * x2, isInverse=True)
            X2 = a1 * DFT(x1, isInverse=True) + a2 * DFT(x2, isInverse=True)
            assert np.allclose(X1, X2)

    def test_IDFTmat(self):
        for N in self.N_list:
            X = np.random.rand(N)
            x = IDFT(X)
            x_mat = np.dot(IDFTmat(N), X)
            assert np.allclose(x, x_mat)

    def test_IDFTmat_is_orthogonal(self):
        for N in self.N_list:
            assert np.allclose(np.dot(IDFTmat(N), IDFTmat(N).H), np.eye(N))
