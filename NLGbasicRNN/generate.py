from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.preprocessing.text import text_to_word_sequence as t2ws
from keras.utils import np_utils
from prepData import Lyrics
from sklearn.feature_extraction.text import CountVectorizer

import csv, getSysArgs
import numpy as np


def main():
    ## To gather training lyric data
    train = Lyrics()

    train.firstSong(getSysArgs.usage(['generate.py',
                                '<lyric_data_file_path>'])[1:])

    ## Length of training sequence
    seqLen = 30

    train.modelData(seqLen)

    ## Build generator model
    model = Sequential()

    model.add(LSTM(256, input_shape = (train.dataX.shape[1],
                                       train.dataX.shape[2])))#,
#                   return_sequences = True))

#    model.add(Dropout(0.2))
#    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(train.dataY.shape[1], activation = 'softmax'))

    ## Model weights file
    fName = '1layer-weights-improvement-19-4.1462.hdf5'
#    fName = 'weights-improvement-49-2.2582.hdf5'

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
        initNumSeq.append(train.words.index(word))

    for i in range(10):
        ## Generation initialization for model
        genX = np.array(initNumSeq)

        genX = genX.reshape((1, genX.shape[0], 1))
        genX = genX / float(len(train.words))

        ## Predicted word
        pred = model.predict(genX, verbose = 0)

        ## Next word index
        nextI = np.argmax(pred)

        genTxt += ' ' + train.words[nextI]
        initNumSeq.append(nextI)
        initNumSeq = initNumSeq[1:]

    print genTxt

    return

if __name__ == '__main__':
    main()
