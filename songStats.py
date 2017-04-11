from keras.preprocessing.text import text_to_word_sequence as t2ws

import csv, getSysArgs
import numpy as np


def main():
    ## CSV file of lyrics
    [fName] = getSysArgs.usage(['songStats.py', '<lyric_data_file_path>'])[1:]

    ## Open CSV file of lyrics
    dataCSV = csv.reader(open(fName, 'rU'))

    ## Column headers
    colNames = dataCSV.next()

    ## To store song statistics (assumes lyrics are in right-most column)
    stats = [colNames[:-1] + ['lyricCharCt', 'lyricWordCt', 'lyricVocabSize']]

    for row in dataCSV:
        ## Lyrics for a song
        lyrics = row[colNames.index('lyrics')]

        ## Word sequence from song lyrics
        lyricSeq = t2ws(lyrics)

        stats.append(row[:-1] + [len(lyrics), len(lyricSeq),
                                 len(set(lyricSeq))])

    ## File to write song statistics
    statsCSV = csv.writer(open('songStats.csv', 'wb', buffering = 0))

    statsCSV.writerows(stats)

    return

if __name__ == '__main__':
    main()
