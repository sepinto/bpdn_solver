import numpy as np

FULL_BLOCK_LENGTH = 512
COMPRESSED_BLOCK_LENGTH = FULL_BLOCK_LENGTH / 4

class CodingParams:
    """A class to hold coding parameters to share across files"""
    pass  # will just add attributes at runtime as needed

def setDecodingParams(coding_params):
    coding_params.full_block_length = FULL_BLOCK_LENGTH
    coding_params.compressed_block_length = COMPRESSED_BLOCK_LENGTH
    coding_params.epsilon = np.power(10.0, -5.0)
    coding_params.mu = 50.0
    coding_params.alpha = 0.1
    coding_params.beta = 0.5
    coding_params.max_iterations = 5
    coding_params.rho = 0.1
    coding_params.alpha_admm = 1.7
    return coding_params