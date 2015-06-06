import numpy as np

def IDFTmat(N):
    """Calculate the NxN orthogonal IDFT matrix using outer products"""
    k = np.linspace(0, N - 1, N)
    return np.exp(2.0 * np.pi * 1j * np.outer(k, k) / float(N)) / np.sqrt(float(N))

def DFT(f, isInverse=False):
    """Calculate the DFT using the fft algorithm"""
    N = len(f)
    if isInverse:
        return np.sqrt(N) * np.fft.ifft(f)
    else:
        return np.fft.fft(f) / np.sqrt(N)

def IDFT(x):
    """Calculate the IDFT using the fft algorithm"""
    return DFT(x, True)