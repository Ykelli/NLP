# NLPProjectGenrePrediction

run with command:
python3 recommendations.py (options)

Needs to be run with at least one of these options: 
-gb (dataset): run genre train and test with selected dataset (basic, full, large)
-gg : get the predicted genre of entered song lyrics
-as : add songs
-sa : add a single song and get recommendations
-sr : get song recommendations for a selected song

The current data is a small subset of the total data due to limits on space, so only contains 100 000 of a possible 210 000 songs.
Won't be able to run readGenreData.py also due to filesize limitations not allowing required documents to be included, but left so it can be seen.
