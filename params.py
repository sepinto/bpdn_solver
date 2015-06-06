import numpy as np

INPUT_FILENAME = "test"
OUTPUT_FILENAME = INPUT_FILENAME + "_out"
PE_THRESHOLD = 50
QUANT_BITS = 31
FULL_BLOCK_LENGTH = 512
COMPRESSED_BLOCK_LENGTH = FULL_BLOCK_LENGTH / 4
np.random.seed()
SEED = np.random.randint(100)
OVERLAP = 0

def setCodingParams(coding_params):
    coding_params.full_block_length = FULL_BLOCK_LENGTH
    coding_params.compressed_block_length = COMPRESSED_BLOCK_LENGTH
    coding_params.seed = SEED
    coding_params.quant_bits = QUANT_BITS
    coding_params.pe_threshold = PE_THRESHOLD
    coding_params.overlap = OVERLAP
    coding_params.bitsPerSample = 16
    return coding_params

def setDecodingParams(coding_params):
    coding_params.epsilon = np.power(10.0, -5.0)
    coding_params.mu = 50.0
    coding_params.alpha = 0.1
    coding_params.beta = 0.5
    coding_params.max_iterations = 5
    coding_params.rho = 0.1
    coding_params.alpha_admm = 1.7
    return coding_params