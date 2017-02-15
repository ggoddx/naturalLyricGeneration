__author__ = 'snehabhadbhade'

import csv, nltk
import numpy as np
from collections import defaultdict
from langdetect import detect, detect_langs
import re



def main():
    ## CSV file of lyrics

    artists = defaultdict()
    years = defaultdict()
    genres = defaultdict()

    regex = re.compile('[^a-zA-Z]')

    cnt = 0

    fName = "/Users/snehabhadbhade/Downloads/lyrics.csv"
    ## Open CSV file of lyrics

    ## Open CSV file of lyrics

    with open(fName) as csvfile:

        reader = csv.DictReader(csvfile)
        for row in reader:

            if cnt>500:
                break
            cnt+=1

        ## Lyrics (song text) in row
            lyrics = row['lyrics']
            flag_en = 0

            if not lyrics:
                continue
            else:
                print(row['index'])
                lyrics = regex.sub(' ', lyrics)
                if not lyrics.strip():
                    continue
                res = detect_langs(lyrics)
                for item in res:
                    if item.lang == 'en':
                        flag_en = 1
                        break

            if flag_en == 0:
                continue

            lyrics = lyrics.rstrip("\n")


        ## Artist in row
            artist = row['artist']

        ## Genre in row
            genre = row['genre']

        ## Year in row
            year = row['year']

            if artist:
                if artist not in artists:
                    artists[artist] = lyrics
                else:
                    artists[artist] += lyrics
            if genre:
                if genre not in genres:
                    genres[genre] = lyrics
                else:
                    genres[genre] += lyrics
            if year:
                if year not in years:
                    years[year] = lyrics
                else:
                    years[year] += lyrics


    with open('genres.txt', 'w') as f_genres:
        for key, value in genres.items():
            row = key + "\t" + value + "\n"
            f_genres.write(row)

    with open('artists.csv', 'w') as f_artists:
        for key, value in artists.items():
            row = key + "\t" + value + "\n"
            f_artists.write(row)

    with open('years.csv', 'w') as f_years:
        for key, value in years.items():
            row = key + "\t" + value + "\n"
            f_years.write(row)

if __name__ == '__main__':
    main()

