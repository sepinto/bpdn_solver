import numpy as np
from src.functions.matrices import RandomSubset

class CSCompressor():
    """A compressed sensing encoder uniquely identified by it's random seed, N, and m"""

    def __init__(self, coding_params):
        np.random.seed(coding_params.seed)
        self.idxs = RandomSubset(coding_params.full_block_length, coding_params.compressed_block_length)

    def compress(self, f):
        return f[self.idxs]