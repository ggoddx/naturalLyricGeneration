from __future__ import print_function
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.utils import np_utils
from pickle import dump
from prepData import Lyrics

import csv, getSysArgs, time
import numpy as np


def main():
    ## The specified group on which to train, lyric data, and GBs of memory
    [fName, groupType, group, memGB] = getSysArgs.usage(
        ['trainRnn.py', '<lyric_data_file_path>', '<group_type>',
         '<group_name>', '<memory_size_limit_in_GB>'])[1:]

    ## Length of training sequence
    seqLen = 30

    ## To gather training lyric data
    train = Lyrics()

    train.lyrics2seqs(groupType, group, [fName], seqLen)
    dump(train, open('./seqs/' + groupType + '-' + group + '-seq.pkl', 'wb'))

    ## Number of epochs for training
    epochs = 50

    ## Vocabulary size
    vocabSize = len(train.words)

    ## Training data chunk size
    #  (size chosen to be the number of data points that can fit into the
    #   user-specified memory limit)
    chunkSize = int(memGB) * 1073741824 / ((seqLen + vocabSize) * 8)

    ## Range of epochs
    epochRng = range(epochs)

    ## Trained weights file from previous iteration
    prevCP = ''

    ## Number of training datapoints for model
    dataPts = 0

    for song in train.lyricSeq:
        dataPts += len(song) - seqLen

    ## Chunks for training model
    chunks = [chunkSize] * (dataPts / chunkSize) + [dataPts % chunkSize]

    if chunks[-1] == 0:
        chunks = chunks[:-1]

    ## Number of chunks
    numChunks = len(chunks)

    ## Build model
    model = Sequential()

    model.add(LSTM(256, input_shape = (seqLen, 1), return_sequences = True)) #1
    model.add(Dropout(0.2))
    model.add(LSTM(256))  #2nd layer
    model.add(Dropout(0.2))
    model.add(Dense(vocabSize, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

    ## Chunk counter
    chunkC = 0

    ## Filename for datapoints
    dpFname = "./datapoints/%s-%s_chunk_%dof%d.npz"

    ## Position in lyric sequence
    seqI = 0

    ## Index of song in review
    songI = 0

    for chunk in chunks:
        print("chunk %d of %d" % (chunkC + 1, numChunks))
        print("starting at song %d of %d at word %d/%d"
              % (songI, len(train.lyricSeq), seqI, len(train.lyricSeq[songI])))

        ## Observations
        dataX = []

        ## Responses
        dataY = []

        ## Number range the size of the given chunk
        chunkRng = range(chunk)

        ## Starting time
        ti = time.time()

        for i in chunkRng:
            print("datapoint %d/%d %ds" % (i + 1, chunk, time.time() - ti),
                  end = '\r')

            if seqI >= len(train.lyricSeq[songI]) - seqLen:
                songI += 1
                seqI = 0

            dataX.append(train.numSeq[songI][seqI:seqI + seqLen])
            dataY.append([0] * vocabSize)
            dataY[-1][train.numSeq[songI][seqI + seqLen]] = 1
            seqI += 1

        print('\n')
        ti = time.time()
        dataX = train.normObs(np.array(dataX, dtype = np.float64),
                              (chunk, seqLen, 1))

        dataY = np.array(dataY, dtype = np.float64)
        print("numpy arrays created in %d seconds" % (time.time() - ti))
        np.savez(dpFname % (groupType, group, chunkC + 1, numChunks),
                 X = dataX, Y = dataY)

        chunkC += 1

    ## Loss Data
    losses = [['Step', 'Loss']]

    ## Counter for loss steps
    lossC = 1

    ## Lowest loss from model training
    minLoss = float('inf')

    for i in epochRng:
        print("Epoch %d of %d" % (i + 1, epochs))
        chunkC = 0

        for chunk in chunks:
            print("chunk %d of %d" % (chunkC + 1, numChunks))
            ti = time.time()
            data = np.load(dpFname % (groupType, group, chunkC + 1, numChunks))
            print("chunk loaded in %d seconds" % (time.time() - ti))

            ## Train model with datapoints and store callbacks history
            cbHist = model.fit(data['X'], data['Y'], epochs = 1,
                               batch_size = 64)

            ## Loss of current chunk's training
            currLoss = cbHist.history['loss'][0]

            if currLoss < minLoss:
                ## Checkpoint filename
                fNameCP = "./weights/%s-weights-improvement-%.4f-epoch_%dof%d-chunk_%dof%d.hdf5" % (group, currLoss, i + 1, epochs, chunkC + 1, numChunks)

                model.save(fNameCP)
                minLoss = currLoss

            losses.append([lossC, currLoss])
            lossC += 1
            chunkC += 1

        print("end of epoch %d" % (i + 1))

    ## File to write loss data
    lossCSV = csv.writer(open('losses.csv', 'wb', buffering = 0))

    lossCSV.writerows(losses)

    return

if __name__ == '__main__':
    main()
