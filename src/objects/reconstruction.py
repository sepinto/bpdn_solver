import numpy as np

from src.functions.window import Window, WindowPower
from src.functions.dct import IDCT
from src.functions.dft import DFT
from src.functions.psychoac import HalfLengthDFTIntensity, vSPL

class Reconstruction:
    def __init__(self, reconstructed_freq, coding_params, start_idx):
        # DEFINE CODEC CONSTANTS
        self.full_block_length = coding_params.full_block_length
        self.fs = coding_params.sample_rate
        self.num_channels = coding_params.num_channels

        # DEFINE RECONSTRUCTION SPECIFIC DATA
        # Full block length data fields
        self.reconstructed_freq = reconstructed_freq
        self.reconstructed_data = np.zeros_like(reconstructed_freq)
        self.reconstructed_dft = np.zeros_like(self.reconstructed_data, dtype=np.complex)
        self.windowed_data = np.zeros_like(self.reconstructed_data)

        # Single value data fields
        self.start_idx = start_idx
        self.end_idx = start_idx + self.full_block_length

        self.calc_reconstructed_time_data()
        self.calc_reconstructed_dft()
        self.window_data()

    def calc_reconstructed_time_data(self):
        for i in range(self.num_channels):
            self.reconstructed_data[i] = IDCT(self.reconstructed_freq[i])

    def calc_reconstructed_dft(self):
        for i in range(self.num_channels):
            self.reconstructed_dft[i] = DFT(self.reconstructed_data[i])

    def window_data(self):
        for i in range(self.num_channels):
            self.windowed_data[i] = Window(self.reconstructed_data[i])

    def half_length_dft_intensity(self):
        dft_int = np.zeros((self.num_channels, self.full_block_length / 2.0))
        for i in range(self.num_channels):
            dft_int[i] = HalfLengthDFTIntensity(self.reconstructed_dft[i], WindowPower(self.full_block_length))
        return dft_int

    def half_length_dft_spl(self):
        dft_int = self.half_length_dft_intensity()
        dft_spl = np.zeros_like(dft_int)
        for i in range(self.num_channels):
            dft_spl[i] = vSPL(dft_int[i])
        return dft_spl

    def freq_bins(self):
        return np.fft.fftfreq(self.full_block_length, 1.0 / self.fs)

    def time_segment(self):
        return np.linspace(self.start_idx * 1.0 / self.fs, (self.end_idx - 1) * 1.0 / self.fs, self.full_block_length)


