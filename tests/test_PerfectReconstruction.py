import numpy as np
from signals import *
from params import *
from src.objects.cs_file import *
from src.objects.audiofile import *
import pickle

class TestPerfectReconstruction:
    def setUp(self):
        self.num_blocks = 20
        self.f = sparse_file(FULL_BLOCK_LENGTH * self.num_blocks)
        self.wavFileName = "wavs/input/test.wav"
        self.compressedFile = "wavs/csac/test.csac"
        wavFile = PCMFile(self.wavFileName)

        self.coding_params = CodingParams()
        self.coding_params.sample_rate = int(fs)
        self.coding_params.num_channels = 2
        self.coding_params.num_samples = len(self.f)
        self.coding_params.bitsPerSample = 16
        self.coding_params.full_block_length = FULL_BLOCK_LENGTH
        self.coding_params.compressed_block_length = FULL_BLOCK_LENGTH / 64
        self.coding_params.seed = SEED
        self.coding_params.quant_bits = QUANT_BITS
        self.coding_params.pe_threshold = PE_THRESHOLD
        self.coding_params.overlap = 0

        # open the output file
        wavFile.OpenForWriting(self.coding_params)
        for k in range(self.num_blocks):
            data = [self.f[k * FULL_BLOCK_LENGTH:(k+1) * FULL_BLOCK_LENGTH],
                    self.f[k * FULL_BLOCK_LENGTH:(k+1) * FULL_BLOCK_LENGTH]]
            wavFile.WriteDataBlock(data, self.coding_params)

        wavFile.Close(self.coding_params)

    # def test_perfect_reconstruction_without_quantization(self):
    #     # Compress from f directly
    #     csacFile = CSFile(self.compressedFile)
    #     csacFile.OpenForWriting(self.coding_params)
    #
    #     for k in range(self.num_blocks):
    #         data = [self.f[k * FULL_BLOCK_LENGTH:(k+1) * FULL_BLOCK_LENGTH],
    #                 self.f[k * FULL_BLOCK_LENGTH:(k+1) * FULL_BLOCK_LENGTH]]
    #         csacFile.WriteDataBlock(data, self.coding_params)
    #
    #     csacFile.WriteFile()
    #     csacFile.Close(self.coding_params)
    #
    #     # Read From Compressed File
    #     inFile = CSFile(self.compressedFile)
    #     coding_params = inFile.OpenForReading()
    #     coding_params = setDecodingParams(coding_params)
    #
    #     # Read the input file and pass its data to the output file to be written
    #     for k in range(self.num_blocks):
    #         reconstructed_data = inFile.ReadDataBlock(coding_params)
    #         original_data = np.copy(self.f[k * FULL_BLOCK_LENGTH:(k+1) * FULL_BLOCK_LENGTH])
    #         if not np.allclose(original_data, reconstructed_data[0], rtol=np.power(10, -7.0), atol=np.power(10, -7.0)):
    #             pickle.dump(original_data, open( "block_" + str(k) + ".pkl", "wb" ))
    #         assert np.allclose(original_data, reconstructed_data[0], rtol=np.power(10, -7.0), atol=np.power(10, -7.0))
    #         assert np.allclose(original_data, reconstructed_data[1], rtol=np.power(10, -7.0), atol=np.power(10, -7.0))
    #
    #     inFile.Close(coding_params)

    # def test_perfect_reconstruction_from_wav_file(self):
    #     # Compress from f from wav file
    #     wavFile = PCMFile(self.wavFileName)
    #     wavFile.OpenForReading()
    #     self.coding_params.bytesReadSoFar = 0
    #
    #     csacFile = CSFile(self.compressedFile)
    #     csacFile.OpenForWriting(self.coding_params)
    #
    #     k = 0
    #     wavData = np.zeros(self.coding_params.num_samples)
    #     while True:
    #         data = wavFile.ReadDataBlock(self.coding_params)
    #         if not data:
    #             break
    #         wavData[k * FULL_BLOCK_LENGTH:(k+1) * FULL_BLOCK_LENGTH] = data[0].copy()
    #         k += 1
    #         csacFile.WriteDataBlock(data, self.coding_params)
    #
    #     csacFile.WriteFile()
    #     wavFile.Close(self.coding_params)
    #     csacFile.Close(self.coding_params)
    #
    #     # Read From Compressed File
    #     inFile = CSFile(self.compressedFile)
    #     coding_params = inFile.OpenForReading()
    #     coding_params = setDecodingParams(coding_params)
    #
    #     # Read the input file and pass its data to the output file to be written
    #     for k in range(self.num_blocks):
    #         reconstructed_data = inFile.ReadDataBlock(coding_params)
    #         original_data = wavData[k * FULL_BLOCK_LENGTH:(k+1) * FULL_BLOCK_LENGTH]
    #         print "max diff is " + str(np.max(np.absolute(original_data - reconstructed_data)))
    #         assert np.allclose(original_data, reconstructed_data[0], rtol=np.power(10, -4.0), atol=np.power(10, -4.0))
    #         assert np.allclose(original_data, reconstructed_data[1], rtol=np.power(10, -4.0), atol=np.power(10, -4.0))
    #
    #     inFile.Close(coding_params)


