from src.objects.reconstruction import *
from src.objects.cs_reconstructor import *

class Decoder:
    def __init__(self, coding_params):
        self.reconstructor = CSReconstructor(coding_params)

        # State Variables
        self.current_idx = -1 * coding_params.full_block_length * coding_params.overlap
        self.prior_reconstruction = Reconstruction(
            np.zeros((coding_params.num_channels, coding_params.full_block_length)),
            coding_params, 0)
        self.current_reconstruction = Reconstruction(
            np.zeros((coding_params.num_channels, coding_params.full_block_length)),
            coding_params, 0)

    # TODO: Reconstruct differently if transient
    def reconstruct(self, compressed_data, coding_params):
        reconstructed_freq = np.zeros((coding_params.num_channels, coding_params.full_block_length))
        for i in range(coding_params.num_channels):
            reconstructed_freq[i] = self.reconstructor.reconstruct(compressed_data[i])
        return Reconstruction(reconstructed_freq, coding_params, self.current_idx)

    def next_reconstruction(self, reconstruction, coding_params):
        self.prior_reconstruction = self.current_reconstruction
        self.current_reconstruction = reconstruction
        self.current_idx += coding_params.full_block_length * (1 - coding_params.overlap)

    def overlap_and_add(self, coding_params):
        o_and_a = []
        for i in range(coding_params.num_channels):
            first_idx = coding_params.full_block_length * coding_params.overlap
            second_idx = coding_params.full_block_length * (1.0 - coding_params.overlap)

            prior_section = self.prior_reconstruction.windowed_data[i][second_idx:]

            overlapped_section = self.current_reconstruction.windowed_data[i][0:first_idx]
            unoverlapped_section = self.current_reconstruction.windowed_data[i][first_idx:second_idx]

            o_and_a.append(np.concatenate((prior_section + overlapped_section, unoverlapped_section)))
        return o_and_a