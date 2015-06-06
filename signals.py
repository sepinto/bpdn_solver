#! /usr/bin/python
from src.objects.pcmfile import *  # to get access to WAV file handling
from params import *
from src.functions.matrices import RandomSubset
from src.functions.dct import IDCT

fs = 48000.0

def sparse_block(block_length, num_nonzero):
    nonzero_idxs = RandomSubset(block_length, num_nonzero)
    vec = np.zeros(block_length)
    vec[nonzero_idxs] = np.sqrt(block_length / 2) * np.random.rand(num_nonzero)
    return IDCT(vec)

def sparse_file(file_length):
    blocks = int(file_length / FULL_BLOCK_LENGTH)
    remainder = file_length - blocks * FULL_BLOCK_LENGTH
    signal = np.array([])
    for i in range(blocks):
        signal = np.concatenate((signal, sparse_block(FULL_BLOCK_LENGTH, 1)))
    if remainder > 0:
        return np.concatenate((signal, sparse_block(remainder, 1)))
    else:
        return signal

def genSparseDCT(length_in_seconds, inFileName):
    print str(length_in_seconds) + " sec of test signal being written to input.wav..."
    outFile = PCMFile(inFileName)

    N = length_in_seconds * fs
    coding_params = CodingParams()
    coding_params.bitsPerSample = 16
    coding_params.num_channels = 2
    coding_params.sample_rate = fs
    coding_params.num_samples = N

    outFile.OpenForWriting(coding_params)
    outFile.WriteFileHeader(coding_params)

    data = [sparse_file(N), sparse_file(N)]

    outFile.WriteDataBlock(data, coding_params)
    outFile.Close(coding_params)

    print "...Done"

# def sinusoidSignal(N):
#     fc_vals = np.array([440.0, 550.0, 660.0, 880.0, 4400.0, 8800.0])
#     a0 = 0.55
#     a1 = 0.2
#     a2 = 0.11
#     a3 = 0.08
#     a4 = 0.03
#     a5 = 0.01
#     n = np.linspace(0, N - 1, N)
#     x = a0 * np.cos(2 * np.pi * fc_vals[0] * n / fs) + a1 * np.cos(2 * np.pi * fc_vals[1] * n / fs) + \
#         a2 * np.cos(2 * np.pi * fc_vals[2] * n / fs) + a3 * np.cos(2 * np.pi * fc_vals[3] * n / fs) + \
#         a4 * np.cos(2 * np.pi * fc_vals[4] * n / fs) + a5 * np.cos(2 * np.pi * fc_vals[5] * n / fs)
#     return x
#
#
# def genTestWav(length_in_seconds, inFileName):
#     print str(length_in_seconds) + " sec of test signal being written to input.wav..."
#     outFile = PCMFile(inFileName)
#
#     N = length_in_seconds * fs
#     coding_params = CodingParams()
#     coding_params.bitsPerSample = 16
#     coding_params.num_channels = 2
#     coding_params.sample_rate = fs
#     coding_params.num_samples = N
#
#     outFile.OpenForWriting(coding_params)
#     outFile.WriteFileHeader(coding_params)
#
#     data = [sinusoidSignal(N), sinusoidSignal(N)]
#
#     outFile.WriteDataBlock(data, coding_params)
#     outFile.Close(coding_params)
#
#     print "...Done"


if __name__ == "__main__":
    genSparseDCT(FULL_BLOCK_LENGTH * 10 / fs, "wavs/input/test.wav")
