import numpy as np
from scipy.fftpack import dct

def IDCTmat(N):
    """Calculate the NxN orthogonal IDCT matrix using outer products"""
    # http://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-III
    k = np.linspace(0, N - 1, N)
    scale = np.sqrt(2.0 / N) * np.ones(N)
    scale[0] = np.sqrt(1.0 / N)
    return np.outer(np.ones(N), scale) * np.cos(np.pi * np.outer(k + 0.5, k) / N)

def DCT(f, isInverse=False):
    type = 3 if isInverse else 2
    return dct(f, type=type, norm='ortho')

def IDCT(x):
    return DCT(x, True)
