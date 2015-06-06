import numpy as np

from src.functions.window import SinePower


SPL_UPPER_LIMIT = 116
SPL_LOWER_LIMIT = -30


def SPL(intensity):
    """
    Returns the SPL corresponding to intensity (in units where 1 implies 96dB)
    """
    # Intensity of corresponds to -30 dB
    spl = 96.0 + 10.0 * np.log10(intensity + 1e-12)
    if spl < -30: return -30
    return spl


def vSPL(intensity):
    """
    Returns the SPL corresponding to intensity (in units where 1 implies 96dB)
    """
    spl = 96 + 10 * np.log10(intensity + 1e-12)
    spl[spl < SPL_LOWER_LIMIT] = SPL_LOWER_LIMIT
    # TODO: what's a hard upper limit here?
    spl[spl > SPL_UPPER_LIMIT] = SPL_UPPER_LIMIT
    return spl


def Intensity(spl):
    """
    Returns the intensity (in units of the reference intensity level) for SPL spl
    """
    return np.power(10.0, (spl - 96.0) / 10.0)


def HalfLengthDFTIntensity(windowed_dft, window_power):
    N = len(windowed_dft)
    half_dft = windowed_dft[0:N / 2]
    return np.real(4 * half_dft * np.conj(half_dft) / ((N / 2) ** 2 * window_power))


def Thresh(f):
    """Returns the threshold in quiet measured in SPL at frequency f (in Hz)"""
    scaled_f = np.maximum(f, 10) * 0.001  # Limit to 10 Hz - 20 kHz and formula is even
    thresh = 3.64 * np.power(scaled_f, -0.8) - 6.5 * np.exp(-0.6 * np.power(scaled_f - 3.3, 2)) + 0.001 * np.power(
        scaled_f, 4)
    thresh[thresh < SPL_LOWER_LIMIT] = SPL_LOWER_LIMIT
    thresh[thresh > SPL_UPPER_LIMIT] = SPL_UPPER_LIMIT
    return thresh


def Thresh_Int(f):
    """Returns the threshold in quiet measured in Intensity at frequency f (in Hz)"""
    return Intensity(Thresh(f))


def Thresh_Mag(f):
    """Returns the threshold in quiet measured in Real Magnitude at frequency f (in Hz)"""
    N = len(f)
    return np.sqrt((N ** 2 / 4) * Thresh_Int(f))


def Bark(f):
    """Returns the bark-scale frequency for input frequency f (in Hz) """
    scaled_f = f * 0.001
    return 13 * np.arctan(0.76 * scaled_f) + 3.5 * np.arctan(np.power(scaled_f / 7.5, 2))


class Masker:
    """ 
    a Masker whose masking curve decreases linearly in Bark beyond 0.5 Bark from the 
    masker frequency
    """

    def __init__(self, f, SPL, isTonal=True):
        """
        initialized with the frequency and SPL of a masker and whether or not 
        it is Tonal
        """
        self.f = f
        self.z = Bark(f)
        self.SPL = SPL
        self.delta = 15 if isTonal else 5.5
        # Caching some of the computations
        self._slope = (-27 + 0.367 * max(self.SPL - 40.0, 0))

    def IntensityAtFreq(self, freq):
        """The intensity of this masker at frequency freq"""
        return self.vIntensityAtBark(Bark(freq))

    def IntensityAtBark(self, z):
        """The intensity of this masker at Bark location z"""
        dz = z - self.z
        if np.absolute(dz) <= 0.5:
            return Intensity(self.SPL - self.delta)
        elif dz > 0.5:
            return Intensity(
                self.SPL - self.delta + (-27.0 + 0.367 * np.maximum(self.SPL - 40, 0)) * (np.absolute(dz) - 0.5))
        else:
            return Intensity(self.SPL - self.delta + -27.0 * (np.absolute(dz) - 0.5))

    def vSPLAtBark(self, zVec):
        """The intensity of this masker at vector of Bark locations zVec"""
        dz = zVec - self.z
        abs_dz = np.absolute(dz) - 0.5;
        out_spl = np.zeros(len(zVec))
        out_spl[dz > 0.5] = np.multiply(self._slope, abs_dz)[dz > 0.5]
        out_spl[dz < -0.5] = -27.0 * abs_dz[dz < -0.5]
        return out_spl + self.SPL - self.delta

    def vIntensityAtBark(self, zVec):
        """The intensity of this masker at vector of Bark locations zVec"""
        return Intensity(self.vSPLAtBark(zVec))


# Default data for 25 scale factor bands based on the traditional 25 critical bands
cbFreqLimits = np.array(
    [0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400,
     7700, 9500, 12000, 15500], dtype='float')


def AssignFreqLinesFromFreqLimits(half_spectrum_length, sample_rate, flimit=cbFreqLimits):
    if half_spectrum_length < 512:
        flimit = cbFreqLimits[::2]

    idxs = np.concatenate(
        (np.rint(flimit * 2 * half_spectrum_length / sample_rate - 0.5), np.array([half_spectrum_length])))
    idxs = idxs.astype('int')
    return idxs[1:] - idxs[:-1]


class ScaleFactorBands:
    """
    A set of scale factor bands (each of which will share a scale factor and a 
    mantissa bit allocation) and associated freq line mappings.
    
    Instances know the number of bands nBands; the upper and lower limits for 
    each band lowerLimit[i in range(nBands)], upperLimit[i in range(nBands)]; 
    and the number of lines in each band nLines[i in range(nBands)] 
    """

    def __init__(self, nLines):
        """
        Assigns freq lines to scale factor bands based on a vector of the number
        of lines in each band. Indeces are inclusive
        """
        self.nLines = nLines
        self.lowerLine = np.zeros(len(nLines))
        for k in range(1, len(nLines)):
            self.lowerLine[k] = self.lowerLine[k - 1] + nLines[k - 1]
        self.upperLine = np.concatenate((self.lowerLine[1:] - 1, np.array([np.sum(nLines) - 1])))
        self.nBands = len(nLines)


global freq_lookup
global bark_lookup
global thresh_lookup
freq_lookup = {}
bark_lookup = {}
thresh_lookup = {}


def CalcPerceptualEntropy(signal_dft_intensity, mask_dft_intensity, fs, sfBands):
    pe = 0
    for k in range(len(sfBands.nLines)):
        signal_energy = sum(signal_dft_intensity[sfBands.lowerLine[k]:sfBands.upperLine[k]])
        mask_energy = sum(mask_dft_intensity[sfBands.lowerLine[k]:sfBands.upperLine[k]])
        pe += sfBands.nLines[k] * np.log2(1 + np.sqrt(signal_energy / mask_energy))
    return pe


def CalcMaskingThreshold(windowed_dft, sample_rate):
    global freq_lookup

    # Constants
    N = len(windowed_dft)
    sine_power = SinePower(N / 2)

    if N not in freq_lookup.keys():
        freq_lookup[N] = np.fft.fftfreq(N, 1.0 / sample_rate)[0:N / 2]

    freq = freq_lookup[N]

    # Calculate FFT and find maskers
    dft_int = HalfLengthDFTIntensity(windowed_dft, sine_power)
    total_mask_spl = MakeMask(dft_int, sample_rate, N, freq)
    return total_mask_spl


def CalcSMRs(windowed_dft, total_mask_spl):
    # Constants
    N = len(windowed_dft)
    hanning_power = SinePower(N)

    dft_int = HalfLengthDFTIntensity(windowed_dft, hanning_power)
    signal_spl = vSPL(dft_int)
    return signal_spl - total_mask_spl


def IdentifyMaskers(fft_int, fs, N):
    maskers = []
    left_avg = (np.append(np.zeros(2), fft_int[:-2]) + np.append(np.zeros(1), fft_int[:-1])) / 2.0
    right_avg = (np.append(fft_int[2:], np.zeros(2)) + np.append(fft_int[1:], np.zeros(1))) / 2.0

    peaks = np.nonzero(np.greater(fft_int, left_avg) * np.greater(fft_int, right_avg).astype('int'))[0]

    for idx in peaks:
        f = idx * float(fs) / N
        maskers.append(Masker(f, SPL(fft_int[idx])))

    for m in maskers:
        print "f: " + str(m.f) + ", spl: " + str(m.SPL)
    return maskers


def MakeMask(fft_int, fs, N, freq):
    left_avg = (np.append(np.zeros(2), fft_int[:-2]) + np.append(np.zeros(1), fft_int[:-1])) / 2.0
    right_avg = (np.append(fft_int[2:], np.zeros(2)) + np.append(fft_int[1:], np.zeros(1))) / 2.0
    peaks = np.nonzero(np.greater(fft_int, left_avg) * np.greater(fft_int, right_avg).astype('int'))[0]

    global bark_lookup
    if N not in bark_lookup.keys():
        bark_lookup[N] = Bark(freq)
    barks = bark_lookup[N]

    global thresh_lookup
    # Using a MAX operation to combine Maskers
    if N not in thresh_lookup.keys():
        thresh_lookup[N] = Thresh(freq)
    total_mask = thresh_lookup[N]
    for idx in peaks:
        f = idx * float(fs) / N
        spl = SPL(fft_int[idx])
        if spl > total_mask[idx]:
            m = Masker(f, spl)
            # Take the maximum of masking curves
            total_mask = np.maximum(m.vSPLAtBark(barks), total_mask)
    total_mask_spl = total_mask
    return total_mask_spl
