import numpy as np
from signals import *
from params import *
from src.functions.dct import *

__author__ = 'spinto'

class TestSignals:
    def setUp(self):
        self.length_in_seconds = FULL_BLOCK_LENGTH * 10 / fs

    def test_sparse_dct(self):
        num_blocks = 10
        file_length = FULL_BLOCK_LENGTH * num_blocks
        f = sparse_file(file_length)
        assert len(f) == FULL_BLOCK_LENGTH * num_blocks
        for k in range(num_blocks):
            data = f[k * FULL_BLOCK_LENGTH:(k+1) * FULL_BLOCK_LENGTH]
            assert(np.sum(np.greater(np.absolute(DCT(data)), 10.0**-10.0)) == 1)
