import random
from sys import argv
from sklearn.exceptions import ConvergenceWarning
import data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import os
import re

#Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import csv

import nltk
nltk.download('punkt')


# Ignore 'future warnings' from the toolkit.
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=ConvergenceWarning)

#
# YOU MUST WRITE THESE TWO FUNCTIONS (train and test)
#
vectorizer = CountVectorizer(analyzer='word', min_df=7, ngram_range=(1,1)) 
genreModel = LogisticRegression()

def trainGenre(genres):
    '''
    Given a list of passages and their known authors, train your Logistic Regression classifier.
    passages: a List of passage pairs (author,text) 
    Returns: void
    '''
    lyrics = []
    Y_train = []
    for genre in genres.keys():
        for lyric in genres[genre]:
            Y_train.append(genre)
            lyrics.append(lyric)
    X_train_counts = vectorizer.fit_transform(lyrics) 
    genreModel.fit(X_train_counts, Y_train)
    pass

def testGenre(lyrics):
    '''
    Given a list of passages, predict the author for each one.
    passages: a List of passage pairs (author,text)
    Returns: a list of author names, the author predictions for each given passage.
    '''
    X_test_counts = vectorizer.transform(lyrics)
    guesses = genreModel.predict(X_test_counts)
    return guesses   

def trainRecoModel(retrain = False):
    '''
    train the recommendation model and save the files, or read if they already exist
    input
    retrain: whether or not to retrain the model
    return
    fedf: dataframe of song information and embeddings
    model: Doc2Vec model of songs
    '''
    modelFile = "./data/Doc2Vec/trainedModel"
    dataFrameFile = "./data/Doc2Vec/dataframe.csv"
    # check if model already exists and read, otherwise train model
    if (os.path.isfile(modelFile) and os.path.isfile(dataFrameFile) and not retrain):
        print("Reading...")
        fedf = pd.read_csv(dataFrameFile, index_col=0, sep=r'\s*,\s*', header=0, encoding='utf-8', engine='python')
        model = Doc2Vec.load(modelFile)
        return fedf, model
    else:
        print("Training...")
        # get dict of all the songs
        allSongs = data.getRecoData()
        k = allSongs.keys()
        # generate dataframe of the information
        d = {'ID' : list(k), 'song' : [allSongs[x][0] for x in k], 'artist': [allSongs[x][1] for x in k], 'lyrics' : [allSongs[x][2] for x in k]}
        df = pd.DataFrame.from_dict(d)
        # tag the lyrics and train the model
        trainText = [TaggedDocument(words=word_tokenize(d.lower()), tags=[str(i)]) for i, d in list(enumerate(df['lyrics'].tolist()))]
        model = Doc2Vec(vector_size=50, min_count=1, epochs=10, dm=0)
        model.build_vocab(trainText)
        model.train(trainText, total_examples=model.corpus_count, epochs=model.epochs)
        # save the model
        model.save(modelFile)
        # get dataframe of the embeddings
        embdf = pd.DataFrame([model.dv[f'{str(i)}'] for i in range(len(df))])
        # combine the dataframe with embeddings with the dataframe with information
        fedf = df.copy()
        for c in embdf.columns:
            fedf[c] = embdf[c]
        # save the dataframe file
        fedf.to_csv(dataFrameFile)
        return fedf, model

def findRecommendations(df, song):
    """
    find song recommendations based on song input
    input
    df: dataframe of songs and embeddings
    song: songname to check
    return
    similar: list of top 10 similar songs
    """
    # get the vector for the song
    try: 
        songVec = np.array(df.loc[df['song'] == song, [str(i) for i in range(50)]].values.tolist()[0])
    except Exception as e:
        try:
            songVec = np.array(df.loc[df['song'] == song, [i for i in range(50)]].values.tolist()[0])
        except Exception as e:
            print("Song does not exist in current data")
            return 0
        # except Exception as e:
        #     print("Song and artist combination does not exist in current data")
        #     return 0

    # get the vectors and ids for the other songs
    try:
        l = ["ID"] + [str(i) for i in range(50)]
        otherVecs = df.loc[df['song'] != song, l].values.tolist()
    except Exception as e:
        l = ["ID"] + [i for i in range(50)]
        otherVecs = df.loc[df['song'] != song, l].values.tolist()
    sims = []
    # for all of the song vecs calculate the similarity
    for vec in otherVecs:
        id = vec[0]
        vec = np.array(vec[1:])
        sim = np.dot(songVec, vec)/(np.linalg.norm(songVec)*np.linalg.norm(songVec))
        sims.append((id, sim))
    # sort the similarities by the cosine similarity
    sims.sort(key = lambda x: x[1], reverse=True)
    # grab the top 10 songs and the names and artist for each
    sims = sims[:10]
    ids = [sim[0] for sim in sims]
    similar = df.loc[df['ID'].isin(ids), ["song", "artist"]]
    return similar   

def convertLyrics(lyrics):
    '''
    convert a plain set of lyrics to the bag of words format
    '''
    lyricsOut = []
    # retrieve the bag of words included words
    wordList = data.getWords()
    # remove any special characters and change to lower case
    lyrics = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", lyrics)
    lyrics = lyrics.lower()
    lyrics = lyrics.split(" ")
    # iterate through the words in the bag of words, check if in the lyrics and add to the list
    for word in wordList:
        if word in lyrics:
            for x in range(lyrics.count(word)):
                lyricsOut.append(word)
    # return the lyrics
    return ' '.join(lyricsOut)

def getGenre():
    '''
    get the genre estimation of the provided song
    '''
    print("*****get genre*****")
    # get user to input song lyrics
    songLyrics = input("Song Lyrics(enter a blank line when done): ")
    string = input("")
    while(string != ''):
        songLyrics = songLyrics + ' ' + string
        string = input("")
    lyrics = convertLyrics(songLyrics)
    # train the genres model
    print("Training...")
    trainGenres = data.trainData("./data/large/train.csv")
    trainGenre(trainGenres)

    # get predicted genre
    X_test = vectorizer.transform([lyrics])
    guess = genreModel.predict(X_test)
    print(guess[0])

def runGenreTrainTest(gTestName):
    '''
    method to run the train and test the genre model
    input
    gTestName: test files to run the test on
    '''
    print("*****genre train/test*****")
    if(gTestName == "full"):
        trainFile = "./data/full/train.csv"
        testFile = "./data/full/test.csv"
    elif(gTestName == "large"):
        trainFile = "./data/large/train.csv"
        testFile = "./data/large/test.csv"
    else:
        trainFile = "./data/basic/train.csv"
        testFile = "./data/basic/test.csv"

    # train
    print("Training...")
    trainGenres = data.trainData(trainFile)
    trainGenre(trainGenres)

    # test
    print("Testing...")
    testGenres = data.testData(testFile)
    lyrics = []
    gold = []
    for genre in testGenres.keys():
        for lyric in testGenres[genre]:
            lyrics.append(lyric)
            gold.append(genre)
    predicted_labels = testGenre(lyrics)

    # evaluate
    accuracy = data.evaluate(predicted_labels, gold)
    print('Accuracy: %.2f%%\n' % (accuracy))

def addSongs():
    '''
    add songs to the song file for the song recommendations
    return
    retrain: true if songs added
    '''
    retrain = False
    addAnother = input("Would you like to add a song? ")
    while addAnother == "y" or addAnother == "yes":
        retrain = True
        addSong()
        addAnother = input("Would you like to add another song? ")
    return retrain
    
def addSong():
    '''
    helper method to allow breakdown for testing purposes.
    Adds a song to the document for lyrics
    '''
    with open("./data/cleanLyrics/finFullWithoutStopwords.txt", 'a', encoding="utf-8") as f:
        songName = input("Song Name: ")
        artistName = input("Artist Name: ")
        songLyrics = input("Song Lyrics(enter a blank line when done): ")
        string = input("")
        while( string != ''):
            songLyrics = songLyrics + ' ' + string
            string = input("")
        #songID will be name + random float to prevent double IDs for duplicate song titles
        f.writelines(songName[0:2] + str(random.random())[0:16] + '\t' + artistName + '\t' + songName + '\t' + str(convertLyrics(songLyrics)) + '\n')


if __name__ == '__main__':
    #gb, ge
    argslen = len(argv)
    model = None
    if argslen == 1:
        print("usage: args required, use -help for more info")
        exit(1)
    elif argslen >= 2:
        i = 1
        while i < argslen:
            arg = str(argv[i])
            if  arg == "-gb":
                if argslen > 2:
                    i+=1
                    gTestName = str(argv[i])
                else:
                    gTestName = input("Please enter type of test (basic, full, large): ")
                runGenreTrainTest(gTestName)
            #get genre of a song
            elif arg == "-gg":
                getGenre()
            #get song recommendations/add songs
            elif arg == "-sr":
                # give user option to add more song lyrics
                retrain = addSongs()
                # train the model
                df, model = trainRecoModel(retrain)
                # get the songs the user would like recommendations for
                print("*****song recommendations*****")
                song = input("Song Name: ")
                while song != "quit":
                    reco = findRecommendations(df, song)
                    if (not isinstance(reco,int)):
                        print(reco.to_string(index=False))
                    song = input("Song Name: ")
            #add a single song and compare
            elif arg == "-sa":
                if model == None:
                    # train or get model
                    df, model = trainRecoModel()
                    # get song info from user
                    songName = input("Song Name: ")
                    artistName = input("Artist Name: ")
                    songLyrics = input("Song Lyrics(enter a blank line when done): ")
                    string = input("")
                    while( string != ''):
                        songLyrics = songLyrics + ' ' + string
                        string = input("")
                    songLyrics = convertLyrics(songLyrics)
                    # get the calculated embedding for the song
                    embeddings = model.infer_vector([songLyrics])
                    id = songName[0:2] + str(random.random())[0:16]
                    #songID will be name + rnadom float to prevent double IDs for duplicate song titles
                    temp = [id, songName, artistName, songLyrics] + embeddings
                    print(temp)
                    df.append(temp)
                print("*****song recommendations*****")
                reco = findRecommendations(df, songName)
                print(reco.to_string(index=False))
            #a method that adds songs to the fin file until the user wants to stop, just for testing
            elif arg == "-as":
                addSong()
            #print out command line options
            elif arg == "-help":     
                print("""Enter one of the following options: \n
                        -gb (dataset): run genre train and test with selected dataset (basic, full, large)\n
                        -gg : get the predicted genre of entered song lyrics\n
                        -as : add songs\n
                        -sa : add a single song and get recommendations\n
                        -sr : get song recommendations for a selected song""")
            else:
                print('usage: unsupported arg "' + argv[i] + '"')
                print("""Enter one of the following options: \n
                        -gb (dataset): run genre train and test with selected dataset (basic, full, large)\n
                        -gg : get the predicted genre of entered song lyrics\n
                        -as : add songs\n
                        -sa : add a single song and get recommendations\n
                        -sr : get song recommendations for a selected song""")
                exit(2)
            i+=1
    

    print("*****DONE*****")


