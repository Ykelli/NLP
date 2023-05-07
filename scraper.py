import lyricsgenius as lg
import csv
import re



def clean(lyrics):
    s = re.split(" |\n", lyrics)
    tempLyrics = ""
    for i in range(1,len(s)):
        word = s[i]
        tempLyrics += " " + re.sub(r'\W+', '', word)
    return tempLyrics

client_access_token = "T_JxSIu8YFUEi3rDYG_9dOuajcyPXDJV09CtpAg4QFReKrKi_ijn-CSoCRQHflNd"
with open("artists.csv", 'r', encoding='utf-8') as f:
    csv_reader = csv.reader(f, delimiter='\t')
    header = next(csv_reader)
    artistCol = header.index('artist')
    genreCol = header.index('genre')
    artists = dict()
    for row in csv_reader:
        artists[row[artistCol]] = row[genreCol]

api = lg.Genius(client_access_token, timeout=20)
data = []
for artist in artists.keys():
    a = api.search_artist(artist, max_songs=3)
    # while True:
    #     try:
    #         a = api.search_artist(artist, max_songs=3)
    #         break
    #     except:
    #         pass
    for song in a.songs:
        print(song.lyrics)
        data.append([artist, artists[artist], song.title, clean(song.lyrics)])
    break

with open("data.csv", 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(data)
    