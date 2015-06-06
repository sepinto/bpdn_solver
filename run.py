if __name__ == "__main__":
    from params import *
    from src.objects.cs_file import *
    from src.objects.pcmfile import *
    import time
    import sys

    inputWavFile = "wavs/input/%s.wav" % INPUT_FILENAME
    outputWavFile = "wavs/output/%s.wav" % OUTPUT_FILENAME
    compressedFile = "wavs/csac/%s.csac" % OUTPUT_FILENAME

    print "\nTesting the coder:\n   %s \n-> %s \n-> %s" % (inputWavFile, compressedFile, outputWavFile)

    elapsed = time.time()
    for Direction in ("Encode", "Decode"):
        # create the audio file objects
        if Direction == "Encode":
            print "\n\tEncoding input PCM file " + INPUT_FILENAME + ".wav..."
            inFile = PCMFile(inputWavFile)
            outFile = CSFile(compressedFile)
        else:  # "Decode"
            print "\n\tDecoding coded file...\n",
            inFile = CSFile(compressedFile)
            outFile = PCMFile(outputWavFile)

        # open input file
        coding_params = inFile.OpenForReading()
        coding_params = setCodingParams(coding_params)

        # open the output file
        outFile.OpenForWriting(coding_params)  # (includes writing header)

        # Read the input file and pass its data to the output file to be written
        while True:
            data = inFile.ReadDataBlock(coding_params)
            if not data:
                break
            outFile.WriteDataBlock(data, coding_params)
            print ".",  # just to signal how far we've gotten to user

        if Direction == "Encode":
            outFile.WriteFile()
        sys.stdout.flush()
    # end loop over reading/writing the blocks

    # close the files
    inFile.Close(coding_params)
    outFile.Close(coding_params)
# end of loop over Encode/Decode

elapsed = time.time() - elapsed
print "\nDone with Encode/Decode test\n"
print elapsed, " seconds elapsed"



