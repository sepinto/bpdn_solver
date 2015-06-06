"""
csfile.py -- Defines a CSFile class to handle reading and writing audio
data to an audio file holding data compressed using a DFT-based perceptual
compressive sensing audio coding algorithm.  This is a subclass of AudioFile

-----------------------------------------------------------------------
2015 Stephen Pinto & Marina Bosi -- All rights reserved
-----------------------------------------------------------------------

See the documentation of the AudioFile class for general use of the AudioFile
class.

"""

from src.objects.audiofile import CodingParams  # base class
import protobuf.csac_file_pb2 as csac_file_pb2
from src.objects.encoder import *
from src.objects.decoder import *
from params import setDecodingParams


class CSFile:
    def __init__(self, filename):
        self.filename = filename

    def OpenForWriting(self, coding_params):
        self.csac_file = csac_file_pb2.CSACFile()
        self.csac_file.full_block_length = coding_params.full_block_length
        self.csac_file.seed = coding_params.seed
        self.csac_file.sample_rate = coding_params.sample_rate
        self.csac_file.quant_bits = coding_params.quant_bits
        self.csac_file.overlap = coding_params.overlap

        self.channels = []
        for i in range(coding_params.num_channels):
            self.channels.append(self.csac_file.channels.add())

        self.encoder = Encoder(coding_params)

    def OpenForReading(self):
        f = open(self.filename, "rb")
        self.csac_file = csac_file_pb2.CSACFile()
        self.csac_file.ParseFromString(f.read())
        f.close()

        coding_params = CodingParams()
        coding_params.full_block_length = self.csac_file.full_block_length
        coding_params.compressed_block_length = len(self.csac_file.channels[0].blocks[0].samples)
        coding_params.seed = self.csac_file.seed
        coding_params.sample_rate = self.csac_file.sample_rate
        coding_params.num_channels = len(self.csac_file.channels)
        coding_params.num_blocks = len(self.csac_file.channels[0].blocks)
        coding_params.num_samples = coding_params.num_blocks * coding_params.full_block_length / 2.0
        coding_params.quant_bits = self.csac_file.quant_bits
        coding_params.overlap = self.csac_file.overlap

        coding_params = setDecodingParams(coding_params)

        self.decoder = Decoder(coding_params)
        self.curr_block_idx = 0

        return coding_params

    def WriteDataBlock(self, data, coding_params):
        block = self.encoder.overlap(data, coding_params)
        self.encoder.next_block(block, coding_params)
        compressed_vals = self.encoder.compress(coding_params)

        for i in range(coding_params.num_channels):
            # quantized_vals = vQuantizeUniform(compressed_vals[i], coding_params.quant_bits)
            # self.AddBlockToChannel(i, quantized_vals, coding_params)
            block = self.channels[i].blocks.add()
            block.samples.extend(compressed_vals[i].tolist())
            block.transient = bool(self.encoder.detect_transient(coding_params))

    def ReadDataBlock(self, coding_params):
        if self.curr_block_idx >= coding_params.num_blocks:
            return []

        data = np.zeros((coding_params.num_channels, coding_params.compressed_block_length))
        for i in range(coding_params.num_channels):
            # quantized_vals = np.array(self.csac_file.channels[i].blocks[self.curr_block_idx].samples, dtype=np.int32)
            # data[i] = vDequantizeUniform(quantized_vals, coding_params.quant_bits)
            data[i] = np.array(self.csac_file.channels[i].blocks[self.curr_block_idx].samples)
        self.curr_block_idx += 1

        reconstruction = self.decoder.reconstruct(data, coding_params)
        self.decoder.next_reconstruction(reconstruction, coding_params)
        return self.decoder.overlap_and_add(coding_params)

    def WriteFile(self):
        f = open(self.filename, "wb")
        f.write(self.csac_file.SerializeToString())
        f.close()

    def Close(self, coding_params):
        pass



