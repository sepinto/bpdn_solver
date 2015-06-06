from unittest import TestCase

from src.objects.cs_compressor import *
from src.objects.audiofile import CodingParams


__author__ = 'spinto'

class TestCSCompressor(TestCase):
    def setUp(self):
        np.random.seed()
        self.N = np.random.randint(10000)
        self.m = np.random.randint(self.N)
        self.seed = np.random.randint(self.N)

        codingParams = CodingParams()
        codingParams.full_block_length = self.N
        codingParams.compressed_block_length = self.m
        codingParams.seed = self.seed
        self.compressor = CSCompressor(codingParams)

    def test_random_subset(self):
        s = RandomSubset(self.N, self.m)
        assert len(s) == self.m
        assert min(s) >= 0
        assert max(s) < self.N
        assert len(s) == len(set(s))

    def test_compress(self):
        f = np.random.rand(self.N)
        y = self.compressor.compress(f)
        assert len(y) == self.m
        for val in y:
            assert val in f