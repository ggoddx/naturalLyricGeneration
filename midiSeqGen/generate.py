from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
#from keras.preprocessing.text import text_to_word_sequence as t2ws
#from keras.utils import np_utils
from prepData import Bass

#import getSysArgs
import midi
import numpy as np


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

    fName = 'vels-weights-improvement-47-0.0139.hdf5'

    modelVels.load_weights(fName)
    modelVels.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

    modelNotes = Sequential()

    modelNotes.add(LSTM(256, input_shape = (train.notesX.shape[1],
                                            train.notesX.shape[2]),
                        return_sequences = True))

    modelNotes.add(Dropout(0.2))
    modelNotes.add(LSTM(256))
    modelNotes.add(Dropout(0.2))
    modelNotes.add(Dense(train.notesY.shape[1], activation = 'softmax'))

    fName = 'notes-weights-improvement-39-0.6236.hdf5'

    modelNotes.load_weights(fName)
    modelNotes.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

    modelTicks = Sequential()

    modelTicks.add(LSTM(256, input_shape = (train.ticksX.shape[1],
                                            train.ticksX.shape[2]),
                        return_sequences = True))

    modelTicks.add(Dropout(0.2))
    modelTicks.add(LSTM(256))
    modelTicks.add(Dropout(0.2))
    modelTicks.add(Dense(train.ticksY.shape[1], activation = 'softmax'))

    fName = 'ticks-weights-improvement-32-1.3428.hdf5'

    modelTicks.load_weights(fName)
    modelTicks.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

    pattern = midi.read_midifile('bass.mid')

    ticks = []

    notes = []

    vels = []

    for track in pattern:
        for event in track:
            eType = type(event)

            if (eType == midi.events.NoteOnEvent or
                eType == midi.events.NoteOffEvent):
                ticks.append(event.tick)

                noteData = event.data

                notes.append(noteData[0])

                vel = 0

                if noteData[1] > 0:
                    vel = 100

                if eType == midi.events.NoteOffEvent:
                    vel = 0

                vels.append(vel)

    buffer = []

    buffRange = range(200 - len(ticks))

    for i in buffRange:
        buffer.append(-1)

    ticks = buffer + ticks
    notes = buffer + notes
    vels = buffer + vels

    for i in range(1000):
        ticksX = np.array(ticks[-200:])

        ticksX = ticksX.reshape((1, ticksX.shape[0], 1))
        ticksX = ticksX / float(train.maxTick)

        notesX = np.array(notes[-200:])

        notesX = notesX.reshape((1, notesX.shape[0], 1))
        notesX = notesX / float(train.maxNote)

        velsX = np.array(vels[-200:])

        velsX = velsX.reshape((1, velsX.shape[0], 1))
        velsX = velsX / float(train.maxVel)

        predTick = modelTicks.predict(ticksX, verbose = 0)

        nextTick = np.random.choice(np.arange(train.ticksY.shape[1]),
                                    p = predTick[0])

        predNote = modelNotes.predict(notesX, verbose = 0)

        nextNote = np.random.choice(np.arange(train.notesY.shape[1]),
                                    p = predNote[0])

        predVel = modelVels.predict(velsX, verbose = 0)

        nextVel = np.random.choice(np.arange(train.velsY.shape[1]),
                                   p = predVel[0])

#        if vels[-1] == 0:
#            nextVel = 100
#        else:
#            nextVel = 0

        ticks.append(nextTick)
        notes.append(nextNote)
        vels.append(nextVel)

        print 'note', i + 1, 'of 1000'

    genPatt = midi.Pattern()

    genTrack = midi.Track()

    for i in range(len(buffer), len(ticks)):
        genTrack.append(midi.NoteOnEvent(tick = ticks[i], channel = 0,
                                         data = [notes[i], vels[i]]))

    genTrack.append(midi.EndOfTrackEvent(tick = 0))
    genPatt.append(genTrack)
    midi.write_midifile('genBass.mid', genPatt)

    return

if __name__ == '__main__':
    main()
