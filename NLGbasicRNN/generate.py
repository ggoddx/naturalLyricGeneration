from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from pickle import load
from prepData import Lyrics

import csv, getSysArgs
import numpy as np


def main():
    ## The specified group from which to generate lyrics and lyric data
    [groupType, group, seedFile, variation] = getSysArgs.usage(
        ['generate.py', '<group_type>', '<group_name>',
         '<seed_lyrics_file_path>', '<variation_score>'])[1:]

    try:
        variation = int(variation)
    except ValueError:
        variation = float(variation)

    ## Type of variation score provided
    typeVar = type(variation)

    if ((typeVar == int and variation < 1)
        or (typeVar == float
            and (variation <= 0.0 or variation > 1.0))):
        print 'Invalid variation score: ', variation
        print 'Input a float between 0.0 and 1.0 or an int above 1'
        return

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

    seedSeq = ['ppaadd'] * (seqLen - len(seedSeq)) + seedSeq + ['endofline']
    seedSeq = seedSeq[len(seedSeq) - seqLen:]

    if len(seedSeq) < 1:
        print 'Seed lyrics require at least one word\nGiven seed lyrics:'
        print seedLyrics
        return

    ## Number of words in vocabulary
    vocabSize = len(train.words)

    ## Build generator model
    model = Sequential()

    model.add(LSTM(256, input_shape = (seqLen, 1), return_sequences = True)) #1
    model.add(Dropout(0.2))
    model.add(LSTM(256))  #2nd layer
    model.add(Dropout(0.2))
    model.add(Dense(vocabSize, activation = 'softmax'))

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
    genTxt = seedLyrics + ' |\n'

    ## To store prediction distributions
    predDists = np.empty((200, vocabSize))

    ## Vocabulary as a list for use in randomly selecting word
    wordList = [None] * len(train.words)

    for word in train.words:
        wordList[train.words[word]] = word

    for i in range(200):
        ## Generation initialization for model
        genX = train.normObs(np.array(seedNS, dtype = np.float64),
                             (1, seqLen, 1))

        ## Word-prediction distribution
        pred = model.predict(genX, verbose = 0)

        predDists[i] = pred[0]

        if type(variation) == int and variation > vocabSize:
            print variation, 'larger than vocabulary size of', vocabSize
            variation = vocabSize

        if type(variation) == float:
            variation = int(variation * vocabSize)

        ## Indicies of highest probability next words
        topIs = np.argsort(-pred[0])[:variation]

        pred = pred[0][topIs]
        pred = pred / np.sum(pred)

        ## Highest probability words
        topWords = np.array(wordList)[topIs]

        ## Next word based on randomly choosing from word distribution
        next = np.random.choice(topWords, p = pred)

        if next == 'endofsong':
            break
        elif next == 'endofline':
            genTxt += '\n'
        elif next == 'commachar':
            genTxt += ','
        elif next == 'questionmark':
            genTxt += '?'
        else:
            genTxt += ' ' + next

        seedNS.append(train.words[next])
        seedNS = seedNS[1:]

    genTxt = genTxt.replace('endofline', '\n')
    genTxt = genTxt.replace('commachar', ',')
    genTxt = genTxt.replace('questionmark', '?')

    print genTxt

    ## File to write prediction distibutions for each predicted word
    distsCSV = csv.writer(open('predDists.csv', 'wb', buffering = 0))

    distsCSV.writerows(predDists)

    return

if __name__ == '__main__':
    main()
