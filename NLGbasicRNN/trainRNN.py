from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.preprocessing.text import text_to_word_sequence as t2ws
from keras.utils import np_utils
from sklearn.feature_extraction.text import CountVectorizer

import csv, getSysArgs
import numpy as np


def main():
    ## CSV file of lyrics
    [fName] = getSysArgs.usage(['trainRNN.py',
                                '<lyric_data_file_path>'])[1:]

    ## Open CSV file of lyrics
    dataCSV = csv.reader(open(fName, 'rU'))

    ## Column headers
    colNames = dataCSV.next()

    ## One set of lyrics
    lyricData = dataCSV.next()

    ## Song text
    lyrics = lyricData[colNames.index('lyrics')]

    lyrics = lyrics.replace('\n', ' endofline ')
    lyrics = lyrics.replace(',', ' commachar')
    lyrics = lyrics.replace('?', ' questionmark')

    ## Word sequence from lyrics
    lyricSeq = t2ws(lyrics)

    ## Word indicies
    words = list(set(lyricSeq))

    ## Numerical sequence from lyrics
    numSeq = []

    for word in lyricSeq:
        numSeq.append(words.index(word))

    ## Length of training sequence
    seqLen = 30

    ## Training observations
    trainX = []

    ## Training responses
    trainY = []

    ## Range to create training data
    trainRange = range(len(lyricSeq) - seqLen)

    for i in trainRange:
        trainX.append(numSeq[i:i + seqLen])
        trainY.append(numSeq[i + seqLen])

    trainX = np.array(trainX)
    trainX = trainX.reshape(list(trainX.shape) + [1])
    trainX = trainX / float(len(words))
    trainY = np_utils.to_categorical(trainY)

    ## Build generator model
    model = Sequential()

    model.add(LSTM(256, input_shape = (trainX.shape[1], trainX.shape[2]),
                   return_sequences = True))

    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(trainY.shape[1], activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

    ## Checkpoint file path
    chkptFile = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"

    ## Checkpoint
    chkpt = ModelCheckpoint(chkptFile, monitor = 'loss', verbose = 1,
                            save_best_only = True, mode = 'min')

    ## Callbacks list
    cbs = [chkpt]

    model.fit(trainX, trainY, nb_epoch = 50, batch_size = 64,
              callbacks = cbs)

    return

if __name__ == '__main__':
    main()
