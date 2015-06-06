import numpy as np
from src.functions.matrices import *

class TestMatrices():
    def setUp(self):
        self.m_list = [64]#np.power(2, np.linspace(1, 7, 7))
        self.compression_list = [8]#np.power(2, np.linspace(1, 3, 3))
        self.seed = np.random.randint(100)

    def test_left_multiply_phi(self):
        for m in self.m_list:
            for c in self.compression_list:
                n = m * c
                x = np.random.rand(n)
                y = np.random.rand(m)

                phi_indices = RandomSubset(n, m)
                phi = create_Phi(n, phi_indices)

                assert(np.allclose(left_multiply_phi(x, phi_indices, n), np.dot(phi, x)))
                assert(np.allclose(left_multiply_phi(y, phi_indices, n, transpose=True), np.dot(phi.transpose(), y)))

    def test_Phi_A_PhiT(self):
        for m in self.m_list:
            for c in self.compression_list:
                n = m * c
                A = np.random.randn(n, n)

                phi_indices = RandomSubset(n, m)
                phi = create_Phi(n, phi_indices)

                assert(np.allclose(Phi_A_PhiT(A, phi_indices), np.dot(phi, np.dot(A, phi.transpose()))))

    def test_left_multiply_psi(self):
        for m in self.m_list:
            for c in self.compression_list:
                n = m * c
                x = np.random.rand(n)
                psi = create_Psi(n)

                assert np.allclose(left_multiply_psi(x), np.dot(psi, x))
                assert np.allclose(left_multiply_psi(x, transpose=True), np.dot(psi.transpose(), x))

    def test_Psi_diag_PsiT(self):
        for m in self.m_list:
            for c in self.compression_list:
                n = m * c
                diag = np.random.randn(n)
                psi = IDCTmat(n)
                assert(np.allclose(Psi_diag_PsiT(diag), np.dot(psi, np.dot(np.diag(diag), psi.transpose()))))

    def test_left_multiply_Z(self):
        for m in self.m_list:
            for c in self.compression_list:
                n = m * c
                x = np.random.rand(n)
                y = np.random.rand(m)
                phi_indices = RandomSubset(n, m)

                Z = create_Z(phi_indices, n)

                assert np.allclose(left_multiply_Z(x, phi_indices, n), np.dot(Z, x))
                assert np.allclose(left_multiply_Z(y, phi_indices, n, transpose=True), np.dot(Z.transpose(), y))

    def test_Z_diag_ZT(self):
        for m in self.m_list:
            for c in self.compression_list:
                n = m * c
                diag = np.random.randn(n)
                phi_indices = RandomSubset(n, m)
                Z = create_Z(phi_indices, n)
                assert(np.allclose(Z_diag_ZT(diag, phi_indices), np.dot(Z, np.dot(np.diag(diag), Z.transpose()))))
