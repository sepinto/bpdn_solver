import numpy as np

from src.functions.window import Window, WindowPower
from src.functions.dft import DFT
from src.functions.psychoac import CalcMaskingThreshold, CalcSMRs, CalcPerceptualEntropy, AssignFreqLinesFromFreqLimits, \
    ScaleFactorBands, HalfLengthDFTIntensity, Intensity, vSPL


class Block:
    def __init__(self, unwindowed_data, coding_params, start_idx):
        # DEFINE CODEC CONSTANTS
        self.full_block_length = coding_params.full_block_length
        self.fs = coding_params.sample_rate
        self.num_channels = coding_params.num_channels
        self.sfBands = ScaleFactorBands(AssignFreqLinesFromFreqLimits(self.full_block_length / 2, self.fs))

        # DEFINE BLOCK SPECIFIC DATA
        # Full block length data fields
        self.data = unwindowed_data
        self.windowed_data = np.zeros_like(self.data)
        self.windowed_dft = np.zeros((self.num_channels, self.full_block_length), dtype=np.complex_)

        # Half block length data fields
        # Since signal is real, we only deal with a half spectrum covering freq = [0, fs/2)
        self.mask_spl = np.zeros((self.num_channels, self.full_block_length / 2))
        self.smr_spl = np.zeros((self.num_channels, self.full_block_length / 2))

        # Single value data fields
        self.perceptual_entropy = np.zeros(self.num_channels)
        self.start_idx = start_idx
        self.end_idx = start_idx + self.full_block_length

        # CALCULATE BLOCK SPECIFIC DATA FIELDS (all by channel)
        self.window_data()
        self.calc_windowed_dft()
        self.calc_mask_threshold()
        self.calc_smr()
        self.calc_perceptual_entropy()

    def window_data(self):
        for i in range(self.num_channels):
            self.windowed_data[i] = Window(self.data[i])

    def calc_windowed_dft(self):
        for i in range(self.num_channels):
            self.windowed_dft[i] = DFT(self.windowed_data[i])

    def calc_mask_threshold(self):
        for i in range(self.num_channels):
            self.mask_spl[i] = CalcMaskingThreshold(self.windowed_dft[i], self.fs)

    def calc_smr(self):
        for i in range(self.num_channels):
            self.smr_spl[i] = CalcSMRs(self.windowed_dft[i], self.mask_spl[i])

    def calc_perceptual_entropy(self):
        dft_intensity = self.half_length_dft_intensity()
        for i in range(self.num_channels):
            mask_dft_intensity = Intensity(self.mask_spl[i])
            self.perceptual_entropy[i] = CalcPerceptualEntropy(dft_intensity[i], mask_dft_intensity, self.fs,
                                                               self.sfBands)

    def half_length_dft_intensity(self):
        dft_int = np.zeros_like(self.mask_spl)
        for i in range(self.num_channels):
            dft_int[i] = HalfLengthDFTIntensity(self.windowed_dft[i], WindowPower(self.full_block_length))
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


