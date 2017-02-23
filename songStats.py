import csv, getSysArgs
import numpy as np

def main():
    ## CSV file of lyrics
    [fName] = getSysArgs.usage(['songStats.py', '<lyric_data_file_path>'])[1:]

    ## Open CSV file of lyrics
    dataCSV = csv.reader(open(fName, 'rU'))

    ## Column headers
    colNames = dataCSV.next()

    ## To store song statistics
    stats = [colNames[:-1] + ['lyricCharCt']]

    for row in dataCSV:
        stats.append(row[:-1] + [len(row[-1])])

    ## File to write song statistics
    statsCSV = csv.writer(open('songStats.csv', 'wb', buffering = 0))

    statsCSV.writerows(stats)

    return

if __name__ == '__main__':
    main()
