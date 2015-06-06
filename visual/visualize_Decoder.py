import matplotlib.animation as animation

from visual_params import *
from src.objects.cs_file import *
from src.objects.pcmfile import *
from visualize_Encoder import update_freq_lines
from src.functions.dct import DCT

def update_reconstructions(i):
    global inFile, csacFile, outFile, coding_params, num_blocks, num_samples, encoding_codingParams, encoding_csacFile
    global light_gray_time_line, dark_gray_time_line, blue_time_line, black_time_line, gray_freq_line, black_freq_line, black_dct_line

    # Constants for syntactical convenience
    N = coding_params.full_block_length
    fs = coding_params.sample_rate
    overlap = coding_params.overlap

    # Reset files if necessary
    if i == 0:
        csacFile = CSFile(compressedFileName)
        outFile = PCMFile(outputFileName)

        encoding_csacFile = CSFile("wavs/csac/tmp.csac")
        inFile = PCMFile(inputFileName)

        # open input file
        coding_params = csacFile.OpenForReading()
        coding_params = setCodingParams(coding_params)
        encoding_codingParams = inFile.OpenForReading()
        encoding_codingParams = setCodingParams(encoding_codingParams)

        # open the output file
        outFile.OpenForWriting(coding_params)  # (includes writing header)
        encoding_csacFile.OpenForWriting(encoding_codingParams)

    # Do the ordinary process
    data = csacFile.ReadDataBlock(coding_params)
    outFile.WriteDataBlock(data, coding_params)

    # Data we're pulling out of the running process
    curr_reconstruction = csacFile.decoder.current_reconstruction
    prior_reconstruction = csacFile.decoder.prior_reconstruction

    # ENCODING FOR COMPARISON
    # Do the ordinary process
    encoding_data = inFile.ReadDataBlock(encoding_codingParams)
    encoding_csacFile.WriteDataBlock(encoding_data, encoding_codingParams)

    # Data we're pulling out of the running process
    curr_block = encoding_csacFile.encoder.current_block
    prior_block = encoding_csacFile.encoder.prior_block

    update_freq_lines(i, prior_block, curr_block, gray_freq_line, gray_freq_dashed, black_dct_line, coding_params, txt=False)
    freq = curr_reconstruction.freq_bins()[0:coding_params.full_block_length / 2]
    current_dft_spl = curr_reconstruction.half_length_dft_spl()
    black_freq_line.set_data(freq, current_dft_spl[0])
    black_dct_line.set_data(np.linspace(1,N,N), DCT(curr_reconstruction.windowed_data[0]))

    # Plot the time lines correctly
    if i == 0:
        time_ax.set_xlim([0, curr_reconstruction.time_segment()[0] + 10 * N * 1.0 / fs])
        black_time_line.set_data(curr_reconstruction.time_segment()[int(N * (1 - overlap)):],
                              curr_reconstruction.windowed_data[0][int(N * (1 - overlap)):])
        # blue_time_line.set_data([], [])
        dark_gray_time_line.set_data([], [])
    elif i == 1:
        black_time_line.set_data(curr_reconstruction.time_segment(), curr_reconstruction.windowed_data[0])
        # blue_time_line.set_data(prior_reconstruction.time_segment()[int(N * (1 - overlap)):],
        #                     prior_reconstruction.windowed_data[0][int(N * (1 - overlap)):])
        dark_gray_time_line.set_data(np.concatenate((prior_reconstruction.time_segment()[int(N * (1 - overlap)):],
                                                     curr_reconstruction.time_segment()[int(N * overlap):int(N * (1 - overlap))])),
                            csacFile.decoder.overlap_and_add(coding_params)[0])
    elif i > 1:
        black_time_line.set_data(curr_reconstruction.time_segment(), curr_reconstruction.windowed_data[0])
        # blue_time_line.set_data(prior_reconstruction.time_segment(), prior_reconstruction.windowed_data[0])

        # Build total line
        new_x = np.concatenate((dark_gray_time_line.get_xdata(),
                                prior_reconstruction.time_segment()[int(N * (1 - overlap)):],
                                curr_reconstruction.time_segment()[int(N * overlap):int(N * (1 - overlap))]))
        new_y = np.concatenate(
            (dark_gray_time_line.get_ydata(), csacFile.decoder.overlap_and_add(coding_params)[0]))
        dark_gray_time_line.set_data(new_x, new_y)

        # freq = curr_reconstruction.freq_bins()[0:coding_params.full_block_length / 2]
        # current_dft_spl = curr_reconstruction.half_length_dft_spl()
        # black_freq_line.set_data(freq, current_dft_spl[0])

    # Set up running time window if we're far enough into the file
    if i > 20:
        time_ax.set_xlim([curr_reconstruction.time_segment()[0] - 5 * N * 1.0 / fs,
                          curr_reconstruction.time_segment()[0] + 5 * N * 1.0 / fs])

    # If on last block, close all files
    if i == (num_blocks - 1):
        csacFile.Close(coding_params)
        outFile.Close(coding_params)
        inFile.Close(encoding_codingParams)
        encoding_csacFile.Close(encoding_codingParams)


if __name__ == "__main__":
    print "\nVisualizing the decoding process for %s" % (compressedFileName)

    # First plot correct time line to compare against reconstruction
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

    light_gray_time_line.set_data(np.linspace(0, float(num_samples - 1) / coding_params.sample_rate, num_samples), full_data)

    # Get num_blocks and total time for big plotting
    csacFile = CSFile(compressedFileName)
    coding_params = csacFile.OpenForReading()
    num_blocks = coding_params.num_blocks
    num_samples = coding_params.num_samples

    # Make another csac file to do encoding processing so we can compare to actual blocks
    global time_ax, freq_ax, encoding_codingParams, encoding_csacFile
    time_ax.set_title('Reconstruction w/ %d Block Size & %d Compressed Size' % (coding_params.full_block_length, coding_params.compressed_block_length), fontsize=25)
    time_ax.legend(['Original', 'Reconstructed', 'Current Recons. Block'])
    freq_ax.set_title('DFT of Current Block', fontsize=25)
    freq_ax.legend(['Original', 'Original Mask', 'Reconstructed'])
    anim = animation.FuncAnimation(fig, update_reconstructions, frames=num_blocks, interval=50)
    plt.show()

    # vid_file = "vids/%s_reconstruct.mp4" % INPUT_FILENAME
    # print "Writing to " + vid_file
    # Breaks for anything below 5 fps
    # anim.save(vid_file, fps=5)

    csacFile.Close(coding_params)
    outFile.Close(coding_params)




