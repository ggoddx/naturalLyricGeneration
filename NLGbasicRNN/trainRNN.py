from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.utils import np_utils
from prepData import Lyrics

import getSysArgs


def main():
    ## To gather training lyric data
    train = Lyrics()

    ## Specified group on which to train and lyric data file
    [fName, groupType, group] = getSysArgs.usage(
        ['trainRnn.py', '<lyric_data_file_path>', '<group_type>',
         '<group_name>'])[1:]

    train.lyrics2seqs(groupType, group, [fName])

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
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

    ## Checkpoint file path
    chkptFile = group + "-2-weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"

    ## Checkpoint
    chkpt = ModelCheckpoint(chkptFile, monitor = 'loss', verbose = 1,
                            save_best_only = True, mode = 'min')

    ## Callbacks list
    cbs = [chkpt]

    print 'numpy array datatype X: ', train.dataX.dtype
    print 'numpy array datatype Y: ', train.dataY.dtype

    model.fit(train.dataX, train.dataY, nb_epoch = 50, batch_size = 64,
              callbacks = cbs)

    return

if __name__ == '__main__':
    main()
