import pickle
import matplotlib.pyplot as plt
from src.functions.dct import *
from src.objects.cs_compressor import *
from src.objects.cs_reconstructor import *
from params import *
from src.objects.audiofile import CodingParams
from signals import *

def reconstruct_from_compressed_size(block, compressed_size, cp):
    cp.compressed_block_length = compressed_size

    compressor = CSCompressor(cp)
    reconstructor = CSReconstructor(cp)

    compressed = compressor.compress(block)
    reconstructed_dct = reconstructor.reconstruct(compressed)
    return reconstructed_dct

def count_percentage_perfect(block, compressed_size, cp):
    cp.compressed_block_length = compressed_size
    perfect_count = 0
    runs = 100
    for k in range(runs):
        np.random.seed()
        cp.seed = np.random.randint(100)
        reconstructed_dct = reconstruct_from_compressed_size(block, compressed_size, cp)
        perfect = np.all(np.less(np.absolute(reconstructed_dct - DCT(block)), 10.0**-8.0))
        perfect_count += 1 if perfect else 0
    return float(perfect_count) / runs


if __name__ == "__main__":
    block = pickle.load(open("bad_block.txt","rb"))

    cp = CodingParams()
    cp.full_block_length = len(block)
    cp.seed = SEED
    cp.sample_rate = fs
    cp = setDecodingParams(cp)

    plt.plot(DCT(block), label="original", color="k")
    comp_sizes = [cp.full_block_length / 32, cp.full_block_length / 64, cp.full_block_length / 128, cp.full_block_length / 256]
    for compressed_size in comp_sizes:
        reconstructed_dct = reconstruct_from_compressed_size(block, compressed_size, cp)
        perfect = np.all(np.less(np.absolute(reconstructed_dct - DCT(block)), 10.0**-8.0))
        plt.plot(reconstructed_dct, label="m=" + str(compressed_size) + (" - PERFECT" if perfect else ""))

        ratio = count_percentage_perfect(block, compressed_size, cp)
        print "Size " + str(compressed_size) + " got perfect reconstruction " + str(100.0 * ratio) + "% of the time"

    plt.title('DCT Domain of Signal & Reconstructions for n = ' + str(cp.full_block_length))
    plt.ylabel('Amplitude')
    plt.xlabel('Index')
    plt.legend()
    plt.show()