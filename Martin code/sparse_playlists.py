import utils_playlists as utils
from statistics import mean
import numpy as np
import time
import pickle
"""
README

This program, alongside utils_playlists.py, creates a similarity matrix
using word vectors and context windows. 

IMPORTANT: It draws data from the full_data.json, meaning that you must 
move this file and utils_playlists.py to a local subdirectory that also
contains full_data.json, since the json cannot be pushed to github. 

Just to clarify, our full_data.json includes 4000 playlists with approximately
not unique 500,000 songs.

This program shows the two most similar songs and two most different songs.
Additionally, it outputs the most similar songs to a particular song, which
can be specified in line 46. THE SONG NAME MUST BE EXACTLY WHAT IS IN THE JSON.

We should probably figure out a way to 1) have a user input a song, and 2)
somehow account for names that aren't exactly the same, such as capitalizations,
or with or without features (some songs include the name and: (feat. artist_name))

The most_similar method takes a long time to run. 
For a truncated amount of 1000 songs, it took the program 35 seconds to run.
For 5000 songs, the program takes 1369 seconds to run.
For 10000 songs, the program takes too long to run, so I didn't do it.

For the 1000 songs run, the average truncated playlist length was 15.3,
the least similar pair of songs: white christmas, drink in my hand
most similar pair of songs: silent night, the christmas song
the most similar songs to Ed Sheeran's Shape of You are:
	I don't wanna live forever (fifty shades darker), Something Like This, 
	Paris, Stay (with Alessia Cara), Issues

For the 5000 song run, the average truncataed playlist length was 33.6
the least similar pair of songs: Lose Control (feat. Ciara & Fat Man Scoop), Deck The Halls
most similar pair of songs: silent night, I'll be home for Christmas
the most similar songs to Ed Sheeran's Shape of You are:
	I don't wanna live forever (fifty shades darker), Something Like This, 
	Paris, Stay (with Alessia Cara), Issues

"""

def main():
	#hyperparameter
	NUM_CLOSEST = 5 #controls the number of closest songs the the song you want
	TRUNC_AMOUNT = 5000 #this controls how many top songs you want to keep in truncation

	start = time.time() #this is for timing purposes

	data = utils.load_corpus()
	 #list of playlists
	# print(playlists[0]['tracks'][0]['track_name'])

	"""
	I want to make a structure where the corpus is represented by each row
	being a playlist, and then each element in that list to be the songs 
	in that list
	"""
	corpus = utils.make_corpus(playlists)

	#avg length of playlists is the avg length of each row of corpus
	avg_playlist_len = utils.avgLength(corpus)
	print('average playlist length: ' + str(avg_playlist_len))

	#each playlist is truncated to include only the top 5000 most common songs
	trunc_corpus, new_counts = utils.trunc_corpus(corpus, utils.counts(corpus), TRUNC_AMOUNT)

	#average length of playlist is now halved, from ~66 to ~33 songs
	trunc_avg_playlist_len = utils.avgLength(trunc_corpus)
	print('average truncated playlists length: ' + str(trunc_avg_playlist_len))

	# NOTE: Window size parameter depends on the average length of the playlist
	WINDOW_SIZE = int(trunc_avg_playlist_len)

	vocab, inverse_vocab = utils.construct_vocab(trunc_corpus)

	#generating lookup table
	lookup_table = utils.word_vectors(trunc_corpus, vocab, WINDOW_SIZE)
	np_lookup_table = np.asarray(lookup_table)
	print(np_lookup_table)

	#new stuff
	playlists = data['playlists']
	song_id_map = utils.make_dic(playlists)
	uri_map = {}
	for k,v in vocab.items():
		uri_map[k] = song_id_map[k]

	np.save('lookup_table', np_lookup_table)
	# np.save('inverse_vocab', inverse_vocab)
	pickle.dump(inverse_vocab, open('inverse_vocab', "wb"))
	pickle.dump(vocab, open('vocab', "wb"))
	pickle.dump(uri_map, open('song_to_uri', "wb"))
	np.save('playlist', trunc_corpus)

	# np.save('song_to_uri', uri_map)

	# NOTE: To find the most similar song to a particular song, edit the second argument in 
	# most_similar to be vocab['NAME OF SONG']. BUT it must be EXACT name.
	
	# similar, largestPair, smallestPair, top_similars = utils.most_similar(lookup_table, vocab['Shape of You'], NUM_CLOSEST)
	
	# print("Least similar pair of songs: ")
	# print(inverse_vocab[largestPair[0]] + ", " + inverse_vocab[largestPair[1]])
	# print("Most similar pair of songs: ")
	# print(inverse_vocab[smallestPair[0]] + ", " + inverse_vocab[smallestPair[1]])
	# print("The most similar songs to the Ed Sheeran's Shape of You are:")
	# for item in top_similars:
	# 	print (inverse_vocab[item])

	end = time.time()
	print("This program ran for this many seconds:")
	print(end-start)



if __name__ == '__main__':
	main()