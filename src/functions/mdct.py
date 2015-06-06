import numpy as np

def MDCTmat(a, b, isInverse=False):
    N = float(a + b)
    n0 = (b + 1) / 2.0

    n = np.linspace(0, N - 1, N)
    k = np.linspace(0, N/2 - 1, N/2)

    if isInverse:
        return 2.0 * np.cos((2 * np.pi / N) * np.outer(n + n0, k + 0.5))
    else:
        return (2.0 / N) * np.cos((2 * np.pi / N) * np.outer(k + 0.5, n + n0))

def IMDCTmat(N):
    return MDCTmat(N/2, N/2, True)

def MDCT(data, a, b, isInverse=False):
    """
    Fast MDCT algorithm for window length a+b following pp. 141-143 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    (and where 2/N factor is included in forward transform instead of inverse)
    a: left half-window length
    b: right half-window length
    """
    N = float(a + b)
    n0 = (b + 1) / 2.0

    n = np.linspace(0, N - 1, N)
    k = np.linspace(0, N - 1, N) if isInverse else np.linspace(0, N/2 - 1, N/2)

    pre_twiddle = np.exp(1j * 2 * np.pi * k * n0 / N) if isInverse else np.exp(-1j * np.pi * n / N)
    post_twiddle = np.exp(1j * np.pi * (n + n0) / N) if isInverse else np.exp(-1j * 2 * np.pi * n0 * (k + 0.5) / N)

    if isInverse:
        return N * np.real(np.fft.ifft(pre_twiddle * np.concatenate((data, -1 * data[::-1])), int(N)) * post_twiddle)
    else:
        fft = np.fft.fft(pre_twiddle * data, int(N))
        return (2.0 / N) * np.real(fft[0:N/2] * post_twiddle)

def IMDCT(data, a, b):
    return MDCT(data, a, b, isInverse=True)