import csv, getSysArgs, nltk
import numpy as np

def main():
    ## CSV file of lyrics
    [fName] = getSysArgs.usage(['readLyrics.py', '<lyric_data_file_path>'])[1:]

    ## Open CSV file of lyrics
    dataCSV = csv.reader(open(fName, 'rU'))

    ## To store lyric statistics
    lyricStats = [['Song', 'No. Lines', 'No. Words']]

    ## To store artists
    artists = {}

    ## To store genres
    genres = {}

    ## To store years
    years = {}

    ## Column headers
    colNames = dataCSV.next()
    print colNames

    for row in dataCSV:
        ## Index of row
        i = row[colNames.index('index')]

        ## Song in row
        song = row[colNames.index('song')]

        ## Lyrics (song text) in row
        lyrics = row[colNames.index('lyrics')]


        lyricStats.append([(song, i), lyrics.count('\n') + 1,
                           len(nltk.word_tokenize(lyrics.decode('utf-8')))])

    ## File to write lyric statistics
    lyricCSV = csv.writer(open('lyricStats.csv', 'wb', buffering = 0))

    lyricCSV.writerows(lyricStats)

    return

    for row in dataCSV:
        ## Artist in row
        artist = row[colNames.index('artist')]

        ## Genre in row
        genre = row[colNames.index('genre')]

        ## Year in row
        year = row[colNames.index('year')]

        if artist not in artists:
            artists[artist] = 0

        if genre not in genres:
            genres[genre] = 0

        if year not in years:
            years[year] = 0

        artists[artist] += 1
        genres[genre] += 1
        years[year] += 1

    artists = np.transpose(np.array([artists.keys(), artists.values()]))
    genres = np.transpose(np.array([genres.keys(), genres.values()]))
    years = np.transpose(np.array([years.keys(), years.values()]))

    ## File to write artists
    artistCSV = csv.writer(open('artistStats.csv', 'wb', buffering = 0))

    ## File to write genres
    genreCSV = csv.writer(open('genreStats.csv', 'wb', buffering = 0))

    ## File to write years
    yearCSV = csv.writer(open('yearStats.csv', 'wb', buffering = 0))

    artistCSV.writerows(artists)
    genreCSV.writerows(genres)
    yearCSV.writerows(years)

    return

if __name__ == '__main__':
    main()
