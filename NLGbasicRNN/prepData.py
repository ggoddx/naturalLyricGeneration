from keras.preprocessing.text import text_to_word_sequence as t2ws
from keras.utils import np_utils

import csv
import numpy as np


## Handles the collection of lyrics from the dataset
class Lyrics:
    ## Establishes Lyrics object
    def __init__(self):
        return

    ## Builds word sequence from first song in the dataset
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

        ## Word sequence from lyrics
        lyricSeq = self.getWordSeq(lyrics)

        ## Word indicies
        self.words = list(set(lyricSeq))

        ## Numerical sequence from lyrics
        numSeq = self.getNumSeq(lyricSeq)

        self.lyricSeq = [lyricSeq]
        self.numSeq = [numSeq]

        return

    ## Converts word sequence into a numerical sequence
    #
    #  @param wordSeq list
    #   list for word sequence
    def getNumSeq(self, wordSeq):
        ## Numerical sequence from lyrics
        numSeq = []

        for word in wordSeq:
            numSeq.append(self.words.index(word))

        return numSeq

    ## Converts text into a word sequence
    #  (changing non-word characters to words, so that model accounts for them)
    #
    #  @param text string
    #   text to convert to word sequence
    def getWordSeq(self, text):
        text = text.replace('\n', ' endofline ')
        text = text.replace(',', ' commachar')
        text = text.replace('?', ' questionmark')

        return t2ws(text)

    ## Creates word sequences from a set of lyrics
    #
    #  @param groupType string
    #   type of group of lyrics for which to create sequences
    #   (current options: year, artist, genre)
    #
    #  @param group string
    #   the specific group of lyrics for which to create sequences
    #
    #  @param fileSpecs list
    #   list of variables needed to find file with classification data
    #   (currently ordered as [filename])
    #
    #  @param seqLen int
    #   length of sequences for model training
    def lyrics2seqs(self, groupType, group, fileSpecs, seqLen):
        ## File specifications
        [fName] = fileSpecs

        ## Open CSV file of lyrics
        dataCSV = csv.reader(open(fName, 'rU'))

        ## Column headers
        colNames = dataCSV.next()

        ## Word sequences from different songs' lyrics
        lyricSeq = []

        for row in dataCSV:
            ## Lyrics for one song
            lyrics = row[colNames.index('lyrics')]

            if (row[colNames.index(groupType)] == group and
                len(t2ws(lyrics)) > 1):
                lyricSeq.append(
                    ['ppaadd'] * (seqLen - 1) + self.getWordSeq(lyrics) + ['endofsong'])

        ## Word indicies
        words = []

        ## Numerical sequence from lyrics
        numSeq = []

        for song in lyricSeq:
            ## Numerical sequence for one song
            songNS = []

            for word in song:
                if word not in words:
                    words.append(word)

                songNS.append(words.index(word))

            numSeq.append(songNS)

        print words
        print 'number of', group, 'songs: ', len(lyricSeq), len(numSeq)
        print 'vocab count: ', len(words)
        self.lyricSeq = lyricSeq
        self.numSeq = numSeq
        self.words = words

        return

    ## Finalize and normalize numerical sequences for use with LSTM RNN model
    #
    #  @param dataX numpy array
    #   observations array before normalization
    #
    #  @param shape list or tuple
    #   shape needed for LSTM RNN model
    def normObs(self, dataX, shape):
        dataX = dataX.reshape(shape)

        return dataX / float(len(self.words))

    ## To create data useable by LSTM RNN models
    #
    #  @param seqLen int
    #   The number of words in sequence used for prediction
    def modelData(self, seqLen):
        ## Observations
        dataX = []

        ## Responses
        dataY = []

        ## Range of songs from word and number index sequences
        songRange = range(len(self.lyricSeq))

        for i in songRange:
            ## Range to create training data
            trainRange = range(len(self.lyricSeq[i]) - seqLen)

            for j in trainRange:
                dataX.append(self.numSeq[i][j:j + seqLen])
                dataY.append(self.numSeq[i][j + seqLen])

        print 'number of training points: ', len(dataX)
        dataX = np.array(dataX)
        dataX = self.normObs(dataX, list(dataX.shape) + [1])
        dataY = np_utils.to_categorical(dataY)

        self.dataX = dataX
        self.dataY = dataY

        return
