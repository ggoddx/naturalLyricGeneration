from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from prepData import Bass


def main():
    train = Bass()

    modelVels = Sequential()

    modelVels.add(LSTM(256, input_shape = (train.velsX.shape[1],
                                            train.velsX.shape[2]),
                        return_sequences = True))

    modelVels.add(Dropout(0.2))
    modelVels.add(LSTM(256))
    modelVels.add(Dropout(0.2))
    modelVels.add(Dense(train.velsY.shape[1], activation = 'softmax'))
    modelVels.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

    ## Checkpoint file path                                                     
    chkptFile = "vels-weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"

    ## Checkpoint                                                               
    chkpt = ModelCheckpoint(chkptFile, monitor = 'loss', verbose = 1,
                            save_best_only = True, mode = 'min')

    ## Callbacks list                                                           
    cbs = [chkpt]

    modelVels.fit(train.velsX, train.velsY, nb_epoch = 50, batch_size = 64,
              callbacks = cbs)

    modelNotes = Sequential()

    modelNotes.add(LSTM(256, input_shape = (train.notesX.shape[1],
                                            train.notesX.shape[2]),
                        return_sequences = True))

    modelNotes.add(Dropout(0.2))
    modelNotes.add(LSTM(256))
    modelNotes.add(Dropout(0.2))
    modelNotes.add(Dense(train.notesY.shape[1], activation = 'softmax'))
    modelNotes.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

    ## Checkpoint file path
    chkptFile = "notes-weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"

    ## Checkpoint
    chkpt = ModelCheckpoint(chkptFile, monitor = 'loss', verbose = 1,
                            save_best_only = True, mode = 'min')

    ## Callbacks list
    cbs = [chkpt]

    modelNotes.fit(train.notesX, train.notesY, nb_epoch = 50, batch_size = 64,
              callbacks = cbs)

    modelTicks = Sequential()

    modelTicks.add(LSTM(256, input_shape = (train.ticksX.shape[1],
                                            train.ticksX.shape[2]),
                        return_sequences = True))

    modelTicks.add(Dropout(0.2))
    modelTicks.add(LSTM(256))
    modelTicks.add(Dropout(0.2))
    modelTicks.add(Dense(train.ticksY.shape[1], activation = 'softmax'))
    modelTicks.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

    ## Checkpoint file path
    chkptFile = "ticks-weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"

    ## Checkpoint
    chkpt = ModelCheckpoint(chkptFile, monitor = 'loss', verbose = 1,
                            save_best_only = True, mode = 'min')

    ## Callbacks list
    cbs = [chkpt]

    modelTicks.fit(train.ticksX, train.ticksY, nb_epoch = 50, batch_size = 64,
              callbacks = cbs)

    return

if __name__ == '__main__':
    main()
