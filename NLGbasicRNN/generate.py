from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from pickle import load
from prepData import Lyrics

import csv, getSysArgs
import numpy as np


def main():
    ## The specified group from which to generate lyrics and lyric data
    [groupType, group, seedFile] = getSysArgs.usage(
        ['generate.py', '<group_type>', '<group_name>',
         '<seed_lyrics_file_path>'])[1:]

    ## Load in seed lyrics
    seed = open(seedFile, 'rU')

    ## Seed lyrics
    seedLyrics = ''

    for line in seed:
        seedLyrics += line

    if seedLyrics[-1] == '\n':  #remove newline char if last char in lyrics
        seedLyrics = seedLyrics[:-1]

    ## Open CSV tracking the best model filenames for various groups
    bestModels = csv.reader(open('bestModels.csv', 'rU'))

    ## Filename of trained model
    modelFile = ''

    for row in bestModels:  #assumes 1st and 2nd cols are groupType and group
        if row[:2] == [groupType, group]:
            modelFile = row[2]
            break

    if modelFile == '':
        print 'No model has been trained for the', groupType, group, 'yet'
        return

    modelFile = './weights/' + modelFile

    ## Length of training sequence
    seqLen = 30

    ## To gather training lyric data
    train = load(open('./seqs/' + groupType + '-' + group + '-seq.pkl', 'rb'))

    ## Seed lyric sequence
    seedSeq = train.getWordSeq(seedLyrics)

    seedSeq = ['ppaadd'] * (seqLen - len(seedSeq)) + seedSeq
    seedSeq = seedSeq[len(seedSeq) - seqLen:]

    if len(seedSeq) < 1:
        print 'Seed lyrics require at least one word\nGiven seed lyrics:'
        print seedLyrics
        return

    ## Build generator model
    model = Sequential()

    model.add(LSTM(256, input_shape = (seqLen, 1), return_sequences = True)) #1
    model.add(Dropout(0.2))
    model.add(LSTM(256))  #2nd layer
    model.add(Dropout(0.2))
    model.add(Dense(len(train.words), activation = 'softmax'))

    model.load_weights(modelFile)
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

    ## Seed numerical sequence
    seedNS = []

    for word in seedSeq:
        if word in train.words:
            seedNS.append(train.words[word])
        else:  #use most-likely word if seed word not in vocabulary
            seedNS.append(
                np.argmax(
                    model.predict(
                        train.normObs(
                            np.array([train.words['ppaadd']]
                                     * (seqLen - len(seedNS)) + seedNS,
                                     dtype = np.float64),
                            (1, seqLen, 1)), verbose = 0)))

    ## Generated text
    genTxt = seedLyrics + ' |'

    for i in range(200):
        ## Generation initialization for model
        genX = train.normObs(np.array(seedNS, dtype = np.float64), (1, seqLen, 1))

        ## Word-prediction distribution
        pred = model.predict(genX, verbose = 0)

        ## Next word based on randomly choosing from word distribution
        next = np.random.choice(train.words.keys(), p = pred[0])

        if next == 'endofsong':
            break

        genTxt += ' ' + next
        seedNS.append(train.words[next])
        seedNS = seedNS[1:]

    genTxt = genTxt.replace(' endofline ', '\n')
    genTxt = genTxt.replace(' commachar', ',')
    genTxt = genTxt.replace(' questionmark', '?')

    print genTxt

    return

if __name__ == '__main__':
    main()
