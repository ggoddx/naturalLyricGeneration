from sklearn.feature_extraction.text import CountVectorizer

import csv, getSysArgs
import langdetect as ld


def main():
    ## To ensure consistent identification of language for lyrics
    ld.DetectorFactory.seed = 0

    ## CSV file of lyrics
    [fName] = getSysArgs.usage(['songStats.py', '<lyric_data_file_path>'])[1:]

    ## Open CSV file of lyrics
    dataCSV = csv.reader(open(fName, 'rU'))

    ## Column headers
    colNames = dataCSV.next()

    ## Index of lyrics column
    lyricI = colNames.index('lyrics')

    ## To store song statistics (assumes lyrics are in right-most column)
    stats = [colNames[:-1] + ['lyricCharCt', 'lyricWordCt', 'lyricVocabSize', 
                              'lines', 'lang']]

    ## To obtain counts of words where strings of non-alphanumeric characters
    #  are not considered words
    tokenize = CountVectorizer().build_analyzer()

    for row in dataCSV:
        print 'Number of song statistics collected: ', row[0]

        ## Song's language
        lang = None

        ## Lyrics for a song
        lyrics = row[lyricI]

        ## Lines in song
        lines = lyrics.count('\n')

        ## Words in song
        words = tokenize(lyrics)

        ## Number of words in song
        wrdCt = len(words)

        if wrdCt < 1:
            lang = 'none'
        else:
            try:
                lang = ld.detect(lyrics.decode('utf-8'))
            except ld.lang_detect_exception.LangDetectException:
                lang = 'none'
                print lyrics

        stats.append(row[:-1] + [len(lyrics), wrdCt, len(set(words)), lines,
                                 lang])

    ## File to write song statistics
    statsCSV = csv.writer(open('songStats.csv', 'wb', buffering = 0))

    statsCSV.writerows(stats)

    return

if __name__ == '__main__':
    main()
