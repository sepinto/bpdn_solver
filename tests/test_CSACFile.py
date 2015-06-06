from unittest import TestCase
from protobuf.csac_file_pb2 import *
import numpy as np

__author__ = 'spinto'

class TestCSCompressor(TestCase):
    def setUp(self):
        max_int_val = np.power(2, 31)

        np.random.seed()
        self.N = np.random.randint(max_int_val)
        self.seed = np.random.randint(max_int_val)
        self.fs = np.random.randint(max_int_val)
        self.quant_bits = np.random.randint(max_int_val)
        self.overlap = np.random.randn()

        # Don't want data to be too big
        self.num_channels = np.random.randint(10)
        self.num_blocks = np.random.randint(1000)
        self.m = np.random.randint(1000)

        self.data_in = np.random.randint(max_int_val, size=(self.num_channels, self.num_blocks, self.m))
        self.data_in = self.data_in.astype(np.int32)
        self.transients_in = np.random.choice([True, False], size=(self.num_channels, self.num_blocks))
        self.transients_in = self.transients_in.astype(np.bool)

        # Write to CSAC File
        self.csac_file_in = CSACFile()
        self.csac_file_in.full_block_length = self.N
        self.csac_file_in.seed = self.seed
        self.csac_file_in.sample_rate = self.fs
        self.csac_file_in.quant_bits = self.quant_bits
        self.csac_file_in.overlap = self.overlap

        for i in range(self.num_channels):
            chan = self.csac_file_in.channels.add()
            for j in range(self.num_blocks):
                block = chan.blocks.add()
                block.samples.extend(self.data_in[i, j].tolist())
                block.transient = bool(self.transients_in[i, j])

        self.filename = "test.csac"
        f = open(self.filename, "wb")
        f.write(self.csac_file_in.SerializeToString())
        f.close()

    def test_read_csac(self):
        f = open(self.filename, "rb")
        csac_file_out = CSACFile()
        csac_file_out.ParseFromString(f.read())
        f.close()

        assert csac_file_out.full_block_length == self.N
        assert csac_file_out.seed == self.seed
        assert csac_file_out.sample_rate == self.fs
        assert csac_file_out.quant_bits == self.quant_bits
        assert np.less(np.absolute(csac_file_out.overlap - self.overlap), 10**-7)

        data_out = np.zeros((self.num_channels, self.num_blocks, self.m), dtype=np.int32)
        transients_out = np.zeros((self.num_channels, self.num_blocks), dtype=np.bool)
        for i in range(self.num_channels):
            chan = self.csac_file_in.channels[i]
            for j in range(self.num_blocks):
                block = chan.blocks[j]
                data_out[i, j] = np.array(block.samples, dtype=np.int32)
                transients_out[i, j] = block.transient

        assert np.all(data_out == self.data_in)
        assert np.all(transients_out == self.transients_in)

