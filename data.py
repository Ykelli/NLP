import csv

def getWords():
    """
    get the list of words for bag of words model
    return: list of the words in order
    """
    # open data file and read the words into a list
    with open('./data/words.csv') as f:
        t = csv.reader(f, delimiter=',')
        words = next(t)
    return words

def testData(songFile = './data/basic/test.csv'):
    '''
    Get the required test data for the model
    input:
    songFile: file name for the test data
    return: 
    genres: dictionary with genre as key and list of lyrics for that genre as data
    '''
    # open the data file
    with open(songFile, newline='', encoding="utf-8") as csvfile:
        genres = dict()
        read = csv.reader(csvfile, delimiter=',',quotechar="\"")
        # get the column indices for the required data
        header = next(read)
        lyricsCol = header.index("lyrics")
        genreCol = header.index("genre")
        # iterate through remaining rows and adding to the dictionary
        for row in read:
            genre = row[genreCol].replace("-", " ").lower()
            lyrics = row[lyricsCol].lower()
            # if genre already exists append lyrics, otherwise create list
            if genre in genres.keys():
                genres[genre].append(lyrics)
            else:
                genres[genre] = [lyrics]
    return genres

def trainData(songFile = './data/basic/train.csv'):
    '''
    Get the required train data for the model
    input:
    songFile: file name for the train data
    return: 
    genres: dictionary with genre as key and list of lyrics for that genre as data
    '''
    # open the data file
    with open(songFile, newline='', encoding="utf-8") as csvfile:
        genres = dict()
        read = csv.reader(csvfile, delimiter=',',quotechar="\"")
        # get the column indices for the required data
        header = next(read)
        lyricsCol = header.index("lyrics")
        genreCol = header.index("genre")
        # iterate through remaining rows and adding to the dictionary
        for row in read:
            genre = row[genreCol].replace("-", " ").lower()
            lyrics = row[lyricsCol].lower()
            # if genre already exists append lyrics, otherwise create list
            if genre in genres.keys():
                genres[genre].append(lyrics)
            else:
                genres[genre] = [lyrics]
    return genres

def getRecoData(fileName = "./data/cleanLyrics/finFullWithoutStopwords.txt"):
    '''
    get the word data for the recommendation model
    '''
    # uncomment below line to run on smaller dataset
    # fileName = "./data/cleanLyrics/fin.txt"
    with open(fileName, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        allSongs = dict()
        for line in lines:
            songInfo = line.split("\t")
            songID = songInfo[0]
            artistName = songInfo[1]
            songName = songInfo[2]
            lyrics = songInfo[3].strip("\n")
            allSongs[songID] = (songName, artistName, lyrics)
    return allSongs

def evaluate(guesses, golds):
    '''
    The length of the two given lists must be equal.
    Evaluate the accuracy of the guesses
    guesses: a list of genre names
    golds: a list of correct genres
    '''
    # Sanity check.
    if len(guesses) != len(golds):
        print('ERROR IN EVALUATE: you gave me', len(guesses), 'guessed labels, but', len(golds), 'songs.')
        return 0.0

    # Compare the guesses with the gold labels.
    numRight = 0
    numWrong = 0
    rights = dict()
    wrongs = dict()
    for guess, gold in zip(guesses,golds):
        if guess not in rights:
            rights[guess] = 0
        if guess not in wrongs:
            wrongs[guess] = 0
        if guess == gold:
            numRight += 1
            rights[guess] = rights[guess] + 1
        else:
            numWrong += 1
            wrongs[guess] = wrongs[guess] + 1
    
    # Compute precision.
    for y in set(golds):
        if not y in rights:
            p = 0
            r = 0
        else:
            p = rights[y] / (rights[y] + wrongs[y])
            r = rights[y] / golds.count(y)
        print(y)
        print('  Precision = %d/%d = %.2f' % (rights[y], (rights[y]+wrongs[y]), p))
        print('  Recall    = %d/%d = %.2f' % (rights[y], golds.count(y), r))

    # Compute accuracy.
    print("Correct:   " + str(numRight))
    print("Incorrect: " + str(numWrong))
    accuracy = numRight / (numRight+numWrong)
    return accuracy * 100.0