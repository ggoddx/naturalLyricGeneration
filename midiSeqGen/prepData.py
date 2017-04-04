from keras.utils import np_utils

import midi, os
import numpy as np


class Bass:
    ## Establishes Bass MIDI object
    def __init__(self):
        rootdir = './bassMIDI/'

        ticksX = []

        ticksY = []

        notesX = []

        notesY = []

        velsX = []

        velsY = []

        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                pattern = midi.read_midifile(rootdir + file)

                seqLen = 200

                songTicks = []

                songNotes = []

                songVels = []

                rangeBuff = range(seqLen - 1)

                for i in rangeBuff:
                    songTicks.append(-1)
                    songNotes.append(-1)
                    songVels.append(-1)

                for track in pattern:
                    for event in track:
                        if type(event) == midi.events.NoteOnEvent:
                            songTicks.append(event.tick)

                            noteData = event.data

                            songNotes.append(noteData[0])

                            vel = 0

                            if noteData[1] > 0:
                                vel = 100

                            songVels.append(vel)

                trainRange = range(len(songTicks) - seqLen)

                for i in trainRange:
                    ticksX.append(songTicks[i:i + seqLen])
                    notesX.append(songNotes[i:i + seqLen])
                    velsX.append(songVels[i:i + seqLen])
                    ticksY.append(songTicks[i + seqLen])
                    notesY.append(songNotes[i + seqLen])
                    velsY.append(songVels[i + seqLen])

            self.maxTick = np.max(ticksX)
            self.maxNote = np.max(notesX)
            self.maxVel = np.max(velsX)

            ticksX = np.array(ticksX)
            notesX = np.array(notesX)
            velsX = np.array(velsX)
            ticksX = ticksX.reshape(list(ticksX.shape) + [1])
            ticksX = ticksX / float(self.maxTick)
            notesX = notesX.reshape(list(notesX.shape) + [1])
            notesX = notesX / float(self.maxNote)
            velsX = velsX.reshape(list(velsX.shape) + [1])
            velsX = velsX / float(self.maxVel)
            ticksY = np_utils.to_categorical(ticksY)
            notesY = np_utils.to_categorical(notesY)
            velsY = np_utils.to_categorical(velsY)

            print ticksX.shape
            print notesX.shape
            print velsX.shape
            print ticksY.shape
            print notesY.shape
            print velsY.shape

            self.ticksX = ticksX
            self.notesX = notesX
            self.velsX = velsX
            self.ticksY = ticksY
            self.notesY = notesY
            self.velsY = velsY

        return
