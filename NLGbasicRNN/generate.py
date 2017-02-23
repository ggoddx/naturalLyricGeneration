from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.preprocessing.text import text_to_word_sequence as t2ws
from keras.utils import np_utils
from prepData import Lyrics

import getSysArgs
import numpy as np


def main():
    ## To gather training lyric data
    train = Lyrics()

    train.lyrics2seqs('artist', 'disclosure',
                      getSysArgs.usage(['generate.py',
                                        '<lyric_data_file_path>'])[1:])

    ## Length of training sequence
    seqLen = 30

    train.modelData(seqLen)

    ## Build generator model
    model = Sequential()

    model.add(LSTM(256, input_shape = (train.dataX.shape[1],
                                       train.dataX.shape[2]),
                   return_sequences = True))

    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(train.dataY.shape[1], activation = 'softmax'))

    ## Model weights file
    fName = 'disclosure-2-weights-improvement-49-0.7023.hdf5'

    model.load_weights(fName)
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

    ## Text to start generation
    initTxt = '''You don't think I ain't gonna see?
I'm too big and strong, made this way for me
Better get on up, 'cause I'm the boss'''

#    initTxt = '''I'll be giving up, oh
#Home is where the heart is
#And I gave it to you in a paper bag
#Even though it's tarnished
#'''

    ## Sequence for generation initialization
    initSeq = Lyrics().getWordSeq(initTxt)

    ## Generated text
    genTxt = initTxt + ' |'

    ## Numerical sequence
    initNumSeq = train.getNumSeq(initSeq)

    for i in range(10):
        ## Generation initialization for model
        genX = np.array(initNumSeq)

        genX = train.normObs(genX, (1, genX.shape[0], 1))

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
