from params import *
import matplotlib.pyplot as plt

global time_ax, freq_ax, gray_time_line, blue_time_line, black_time_line, gray_freq_line, gray_freq_dashed, black_freq_line, black_freq_dashed, text_box, black_dct_line
fig = plt.figure(figsize=(16, 9))
text_box = plt.text(0, 0, '')

time_ax = plt.subplot(3, 1, 1)
time_ax.set_ylim([-1, 1])
time_ax.set_xlabel('Time [s]', fontsize=15)
light_gray_time_line, = time_ax.plot([], [], "#C0C0C0", lw=1)
dark_gray_time_line, = time_ax.plot([], [], "#484848", lw=1)
blue_time_line, = time_ax.plot([], [], "b", lw=2)
black_time_line, = time_ax.plot([], [], "k", lw=2)

freq_ax = plt.subplot(3, 1, 2)
freq_ax.set_xlim([0, 24000])
freq_ax.set_ylim([-40, 100])
freq_ax.set_xlabel('Frequency [Hz]', fontsize=15)
freq_ax.set_ylabel('SPL [dB]', fontsize=15)
gray_freq_line, = freq_ax.plot([], [], color="#b8b8b8", lw=1)
gray_freq_dashed, = freq_ax.plot([], [], linestyle='--', color="#b8b8b8", lw=1)
black_freq_line, = freq_ax.plot([], [], color="k", lw=2)
black_freq_dashed, = freq_ax.plot([], [], linestyle='--', color="k", lw=1.5)

dct_ax = plt.subplot(3, 1, 3)
black_dct_line, = dct_ax.plot([], [], color="k", lw=2)
dct_ax.set_xlim([0, 512])
dct_ax.set_ylim([-5, 5])


def clear_freq_lines():
    global gray_freq_line, gray_freq_dashed, black_freq_line, black_freq_line, black_dct_line
    gray_freq_line.set_data([], [])
    gray_freq_dashed.set_data([], [])
    black_freq_line.set_data([], [])
    black_freq_line.set_data([], [])
    black_dct_line.set_data([], [])


def clear_time_lines():
    global light_gray_time_line, dark_gray_time_line, blue_time_line, black_time_line
    light_gray_time_line.set_dat([], [])
    dark_gray_time_line.set_data([], [])
    blue_time_line.set_data([], [])
    black_time_line.set_data([], [])


def clear_lines():
    clear_freq_lines()
    clear_time_lines()


global inFile, csacFile, outFile, coding_params, num_blocks, num_samples, transients

inputFileName = "../wavs/input/%s.wav" % INPUT_FILENAME
compressedFileName = "../wavs/csac/%s.csac" % OUTPUT_FILENAME
outputFileName = "../wavs/output/%s.wav" % OUTPUT_FILENAME
