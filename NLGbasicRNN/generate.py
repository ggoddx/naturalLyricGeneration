from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.preprocessing.text import text_to_word_sequence as t2ws
from keras.utils import np_utils
from sklearn.feature_extraction.text import CountVectorizer

import csv, getSysArgs
import numpy as np


def main():
    ## Lyrics file
    fName = '../lyrics.csv'

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

    ## Model weights file
    fName = 'weights-improvement-49-2.2582.hdf5'

    model.load_weights(fName)
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

    ## Text to start generation
#    initTxt = "you don't think i ain't gonna see questionmark endofline i'm too big and strong commachar got this deal for me endofline better get on up commachar 'cause i'm the boss"

    initTxt = "oh baby commachar how you doing questionmark endofline you know i'm gonna cut right to the chase endofline some women were made but me commachar myself endofline i like to"

    ## Sequence for generation initialization
    initSeq = t2ws(initTxt)

    ## Generated text
    genTxt = initTxt + ' |'

    ## Numerical sequence
    initNumSeq = []

    for word in initSeq:
        initNumSeq.append(words.index(word))

    for i in range(10):
        ## Generation initialization for model
        genX = np.array(initNumSeq)

        genX = genX.reshape((1, genX.shape[0], 1))
        genX = genX / float(len(words))

        ## Predicted word
        pred = model.predict(genX, verbose = 0)

        ## Next word index
        nextI = np.argmax(pred)

        genTxt += ' ' + words[nextI]
        initNumSeq.append(nextI)
        initNumSeq = initNumSeq[1:]

    print genTxt

    return

if __name__ == '__main__':
    main()
