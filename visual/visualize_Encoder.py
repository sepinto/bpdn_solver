import matplotlib.animation as animation

from visual_params import *
from src.objects.cs_file import *
from src.objects.pcmfile import *
from src.functions.dct import DCT

def update_freq_lines(i, prior_block, curr_block, dft_line, mask_line, dct_line, coding_params, txt=True):
    global text_box

    N = coding_params.full_block_length
    # Plot it all correctly
    freq = curr_block.freq_bins()[0:N / 2]
    current_dft_spl = curr_block.half_length_dft_spl()
    dft_line.set_data(freq, current_dft_spl[0])
    mask_line.set_data(freq, curr_block.mask_spl[0])
    dct_line.set_data(np.linspace(1,N,N), DCT(curr_block.windowed_data[0]))

    if txt:
        # Clear old PE textbox and put up new one
        max_pe_change = np.max(curr_block.perceptual_entropy - prior_block.perceptual_entropy)
        text_box.remove()
        text_box = plt.text(freq[-2], 60, 'Delta PE = ' + str(max_pe_change), verticalalignment='top',
                            horizontalalignment='right', fontsize=20)

def update_time_lines(i, prior_block, curr_block, prior_line, current_line, coding_params):
    global transients

    # Constants for syntactical convenience
    N = coding_params.full_block_length
    fs = coding_params.sample_rate
    overlap = coding_params.overlap

    # Plot it all correctly
    if i == 0:
        time_ax.set_xlim([0, curr_block.time_segment()[0] + 10 * N * 1.0 / fs])
        current_line.set_data(curr_block.time_segment()[int(N * (1 - overlap)):], curr_block.data[0][int(N * (1 - overlap)):])
        prior_line.set_data([], [])
    elif i == 1:
        current_line.set_data(curr_block.time_segment(), curr_block.data[0])
        prior_line.set_data(prior_block.time_segment()[int(N * (1 - overlap)):], prior_block.data[0][int(N * (1 - overlap)):])
    elif i > 1:
        prior_line.set_data(prior_block.time_segment(), prior_block.data[0])
        prior_line.set_color('r' if black_time_line.get_color() == 'r' else 'b')
        current_line.set_data(curr_block.time_segment(), curr_block.data[0])
        current_line.set_color('r' if csacFile.encoder.detect_transient(coding_params) else 'k')
        transients += 1 if csacFile.encoder.detect_transient(coding_params) else 0

    # Set up running time window if we're far enough into the file
    if i > 20:
        time_ax.set_xlim(
            [curr_block.time_segment()[0] - 5 * N * 1.0 / fs, curr_block.time_segment()[0] + 5 * N * 1.0 / fs])


def update_blocks(i):
    global inFile, csacFile, outFile, coding_params, num_blocks, num_samples, transients
    global black_freq_line, black_freq_dashed, blue_time_line, black_time_line, black_dct_line

    # Reset files if necessary
    if i == 0:
        inFile = PCMFile(inputFileName)
        csacFile = CSFile(compressedFileName)

        # open input file
        coding_params = inFile.OpenForReading()
        coding_params = setCodingParams(coding_params)

        # open the output file
        csacFile.OpenForWriting(coding_params)  # (includes writing header)

        # Reset transient count
        transients = 0

    # Do the ordinary process
    data = inFile.ReadDataBlock(coding_params)
    csacFile.WriteDataBlock(data, coding_params)

    # Data we're pulling out of the running process
    curr_block = csacFile.encoder.current_block
    prior_block = csacFile.encoder.prior_block

    update_time_lines(i, prior_block, curr_block, blue_time_line, black_time_line, coding_params)
    update_freq_lines(i, prior_block, curr_block, black_freq_line, black_freq_dashed, black_dct_line, coding_params)

    # If on last block, close all files
    if i == (num_blocks - 1):
        csacFile.WriteFile()
        inFile.Close(coding_params)
        csacFile.Close(coding_params)
        print "Detected a total of %s transients in the file" % str(transients)

if __name__ == "__main__":
    global dark_gray_time_line
    # Get num_blocks and total time for plotting
    inFile = PCMFile(inputFileName)
    coding_params = inFile.OpenForReading()
    coding_params = setCodingParams(coding_params)
    num_blocks = 0
    full_data = []
    while True:
        data = inFile.ReadDataBlock(coding_params)
        if not data:
            break
        full_data = full_data + data[0].tolist()
        num_blocks += 1
    inFile.Close(coding_params)
    num_samples = len(full_data)

    dark_gray_time_line.set_data(np.linspace(0, float(num_samples - 1) / coding_params.sample_rate, num_samples), full_data)

    print "\nVisualizing the encoding process for %s" % (inputFileName)

    anim = animation.FuncAnimation(fig, update_blocks, frames=num_blocks, interval=50)
    # plt.show()

    vid_file = "vids/%s_block.mp4" % INPUT_FILENAME
    print "Writing to " + vid_file
    ## Breaks for anything below 5 fps
    anim.save(vid_file, fps=10)

    inFile.Close(coding_params)
    csacFile.Close(coding_params)




