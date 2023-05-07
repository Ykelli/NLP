import csv
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def convertToLyrics(stop = True):
    '''
    Take id and vectorised lyrics data and convert to string of lyrics.
    '''
    # create set of stopwords that can be removed from the data if set to remove
    if (stop):
        remove = set(stopwords.words('english'))
        outputName = "./data/dataMerging/dataWithoutStopwords.txt"
    else:
        outputName = "./data/dataMerging/dataWithStopwords.txt"

    # open list of words for the vector into a list
    with open('./data/words.csv') as f:
        t = csv.reader(f, delimiter=',')
        words = next(t)
    # print(len(words))
    # open and read the unprocessed data
    fileName = "./data/vectorData/unprocessed.txt"
    with open(fileName, 'r') as f:
        lines = f.readlines()
        data=[]
        # split each line into ids and lyrics
        for line in lines:
            line = line.split(',')
            trackID = line[0]
            l = line[2:]
            lyrics = []
            # split id and count pairs and append words to data
            for pair in l:
                pair = pair.split(":")
                wordID = int(pair[0])-1
                wordCount = int(pair[1])
                if stop:
                    if words[wordID] not in remove:
                        for i in range(wordCount):
                            lyrics.append(words[wordID])
                else:
                    for i in range(wordCount):
                        lyrics.append(words[wordID])
            tempRow = [trackID, ' '.join(lyrics)]
            data.append(tempRow)
        # print the processed lyrics into the file
        with open(outputName, 'w', encoding='UTF8', newline='') as w:
            for row in data:
                temp = str(row[0]) + "\t" + str(row[1]) + "\n"
                w.write(temp)

def convertIDs(stop = True):   
    '''
    convert the ids stored with the lyrics into corresponding artist and song names
    '''
    songInfo = dict()
    # turn the id data into a dictionary with the song id as key with artist, song name pair
    with open("./data/vectorData/idToData.txt", 'r', encoding='utf-8') as iT:
        lines = iT.readlines()
        for line in lines:
            line = line.split("<SEP>")
            songInfo[line[0]] = (line[1], line[2])

    # set relevant file names if stopwords are being included
    if (stop):
        inputName = "./data/dataMerging/dataWithoutStopwords.txt"
        outputName = "./data/cleanLyrics/finFullWithoutStopwords.txt"
    else:
        inputName = "./data/dataMerging/dataWithStopwords.txt"
        outputName = "./data/cleanLyrics/finFullWithStopwords.txt"
    
    # open id lyric data and pair with corresponding artist and song name
    with open(inputName, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        data=[]
        for line in lines:
            line = line.split('\t')
            trackID = line[0]
            l = line[1]
            tempRow = [trackID, songInfo[trackID], l]
            data.append(tempRow)
        # write the data to the required output file
        with open(outputName, 'w', encoding='utf-8', newline='') as w:
            for row in data:
                temp = str(row[0] + "\t" + row[1][0]) + "\t" + str(row[1][1]) + "\t" + str(row[2])
                w.write(temp)

# run the conversion files with stopwords
convertToLyrics(False)
convertIDs(False)
# run conversion files without stopwords
convertToLyrics()
convertIDs()