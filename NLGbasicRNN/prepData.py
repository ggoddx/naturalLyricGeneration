from keras.preprocessing.text import text_to_word_sequence as t2ws
from keras.utils import np_utils

import csv
import numpy as np


## Handles the collection of lyrics from the dataset
class Lyrics:
    ## Establishes Lyrics object
    def __init__(self):
        return

    ## Converts word sequence into a numerical sequence
    #
    #  @param wordSeq list
    #   list for word sequence
    def getNumSeq(self, wordSeq):
        ## Numerical sequence from lyrics
        numSeq = []

        for word in wordSeq:
            numSeq.append(self.words[word])

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

        ## Group column index
        groupI = colNames.index(groupType)

        ## Open CSV file with language of song
        langCSV = csv.reader(open('songStats_20170415.csv', 'rU'))

        ## Language column index
        langI = langCSV.next().index('lang')

        ## Lyric text column index
        lyricI = colNames.index('lyrics')

        ## Word sequences from different songs' lyrics
        self.lyricSeq = []

        ## Word indicies
        self.words = {}

        ## Counter for word indicies
        wrdC = 0

        for row in dataCSV:
            ## Lyrics for one song
            lyrics = row[lyricI]

            if (row[groupI] == group and len(t2ws(lyrics)) > 1 and
                langCSV.next()[langI] == 'en'):
                ## Lyric sequence of song (with padding and end-of-song marker)
                seq = (['ppaadd'] * (seqLen - 1) + self.getWordSeq(lyrics)
                       + ['endofsong'])

                for word in seq:
                    if word not in self.words:
                        self.words[word] = wrdC
                        wrdC += 1

                self.lyricSeq.append(seq)

        ## Numerical sequence from lyrics
        self.numSeq = []#self.getNumSeq(self.lyricSeq)

        for song in self.lyricSeq:
            ## Numerical sequence for one song
            songNS = self.getNumSeq(song)

            self.numSeq.append(songNS)

        print self.words
        print 'number of', group, 'songs: ', len(self.lyricSeq), len(self.numSeq)
        print 'vocab count: ', len(self.words.keys())

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
