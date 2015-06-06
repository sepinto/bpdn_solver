import numpy as np
from src.functions.dct import IDCT, DCT, IDCTmat
from src.functions.dft import IDFT, DFT, IDFTmat
from scipy.linalg import toeplitz, hankel
from scipy.fftpack import dct

def RandomSubset(n, m):
    s = set([])
    while len(s) < m:
        s.add(np.random.randint(0, n))
    return list(s)

######################################## SENSING MATRIX ####################################################

def create_Phi(n, phi_indices):
    m = len(phi_indices)
    phi = np.zeros((m, n))
    I = np.eye(n)
    for i in range(int(m)):
        phi[i] = I[phi_indices[i]]
    return phi

def left_multiply_phi(x, phi_indices, n, transpose=False):
    if transpose:
        y = np.zeros(n)
        y[phi_indices] = x
        return y
    else:
        return x[phi_indices]

def Phi_A_PhiT(A, phi_indices):
    return A[phi_indices][:, phi_indices]

######################################## PERCEPTUAL MATRIX ####################################################

def create_H(mask_dft):
    # Assumed here that mask_dft is full length and symmetric so that H is real
    n = len(mask_dft)
    return np.real(np.dot(IDFTmat(n), np.dot(np.diag(1.0 / mask_dft), np.matrix(IDFTmat(n)).H)))

def left_multiply_H(x, mask_dft, transpose=False):
    if transpose:
        return np.real(DFT(IDFT(x) / mask_dft))
    else:
        return np.real(IDFT(DFT(x) / mask_dft))


######################################## SAMPLING MATRIX ####################################################

def create_Psi(n):
    return IDCTmat(n)

def left_multiply_psi(x, transpose=False):
    return DCT(x) if transpose else IDCT(x)

def Psi_diag_PsiT(diag):
    return np.asarray(np.dot(IDCT(np.diag(diag)).transpose(), create_Psi(len(diag)).transpose()))
    # Y = dct(np.concatenate((diag, [0])), type=1) / float(len(diag))
    # return (toeplitz(Y[0:-1]) + hankel(Y[1:], np.flipud(Y[1:]))) / 2.0


######################################## SUMMARY MATRIX ####################################################

def create_Z(phi_indices, n):
    return np.dot(create_Phi(n, phi_indices), create_Psi(n))

def left_multiply_Z(x, phi_indices, n, transpose=False):
    if transpose:
        return left_multiply_psi(left_multiply_phi(x, phi_indices, n, True), True)
    else:
        return left_multiply_phi(left_multiply_psi(x), phi_indices, n)

def Z_diag_ZT(diag, phi_indices):
    return Phi_A_PhiT(Psi_diag_PsiT(diag), phi_indices)
