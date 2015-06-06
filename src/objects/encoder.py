from src.objects.block import *
from src.objects.cs_compressor import *


class Encoder:
    def __init__(self, coding_params):
        self.compressor = CSCompressor(coding_params)

        # State Variables
        self.current_idx = -1 * coding_params.full_block_length * coding_params.overlap
        self.prior_block = Block(np.zeros((coding_params.num_channels, coding_params.full_block_length)), coding_params,
                                 0)
        self.current_block = Block(np.zeros((coding_params.num_channels, coding_params.full_block_length)),
                                   coding_params, 0)

    def overlap(self, data, coding_params):
        full_block_data = []
        for iCh in range(coding_params.num_channels):
            second_idx = coding_params.full_block_length * (1.0 - coding_params.overlap)
            prior_section = self.current_block.data[iCh][second_idx:]
            full_block_data.append(np.concatenate((prior_section, data[iCh])))
        return Block(full_block_data, coding_params, self.current_idx)

    def next_block(self, block, coding_params):
        self.prior_block = self.current_block
        self.current_block = block
        self.current_idx += coding_params.full_block_length * (1 - coding_params.overlap)

    def detect_transient(self, coding_params):
        is_transient = False
        for i in range(coding_params.num_channels):
            ch_transient = (self.current_block.perceptual_entropy[i] - self.prior_block.perceptual_entropy[
                i]) > coding_params.pe_threshold
            is_transient = is_transient or ch_transient
        return is_transient

    def compress(self, coding_params):
        compressed = np.zeros((coding_params.num_channels, coding_params.compressed_block_length))
        for i in range(coding_params.num_channels):
            compressed[i] = self.compressor.compress(self.current_block.windowed_data[i])
        return compressed