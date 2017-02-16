from keras.preprocessing.text import text_to_word_sequence as t2ws
from keras.utils import np_utils

import csv
import numpy as np


## Handles the collection of lyrics from the dataset
class Lyrics:
    ## Establishes Lyrics object
    def __init__(self):
        return

    ##  Builds word sequence from first song in the dataset
    #
    #  @param fileSpecs list
    #   list of variables needed to find file with classification data
    #   (currently ordered as [filename])
    def firstSong(self, fileSpecs):
        ## File specifications
        [fName] = fileSpecs

        ## Open CSV file of lyrics
        dataCSV = csv.reader(open(fName, 'rU'))

        ## Column headers
        colNames = dataCSV.next()

        ## First song in dataset
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

        self.lyricSeq = lyricSeq
        self.numSeq = numSeq
        self.words = words

        return

    ## To create data useable by LSTM RNN models
    #
    #  @param seqLen int
    #   The number of words in sequence used for prediction
    def modelData(self, seqLen):
        ## Observations
        dataX = []

        ## Responses
        dataY = []

        ## Range to create training data
        trainRange = range(len(self.lyricSeq) - seqLen)

        for i in trainRange:
            dataX.append(self.numSeq[i:i + seqLen])
            dataY.append(self.numSeq[i + seqLen])

        dataX = np.array(dataX)
        dataX = dataX.reshape(list(dataX.shape) + [1])
        dataX = dataX / float(len(self.words))
        dataY = np_utils.to_categorical(dataY)

        self.dataX = dataX
        self.dataY = dataY

        return
