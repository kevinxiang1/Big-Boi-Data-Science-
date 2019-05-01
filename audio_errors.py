from skimage import io
import numpy as np
import random
import numpy.matlib
import math
import csv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import ast
import sqlite3
import hypertools as hyp
import pickle

#import kmeans template

def main():
	"""
	This function loads the data set as a 2D numpy array in the data variable
	"""

	# The following codes loads the data set into a 2D np array called data
	# with open('data/word_sentiment.csv') as words_file:

	# file = open("./previous data/500_songs.txt", "r")
	conn = sqlite3.connect('./FULL DATA/tracks2features_full.db')
	c = conn.cursor()
	c.execute("SELECT * FROM audio_features")
	rows = c.fetchall()
	# vocab = pickle.load(open("vocab", "rb"))

	# new_rows = []
	# for song in rows:
	# 	if song[0] in vocab:
	# 		rows.append(song)

	# my_dict = ast.literal_eval(file.read())
	data1 = []
	data2 = []
	data3 = []
	data4 = []
	data5 = []
	data6 = []

	maxValence = 0
	maxTempo = 0
	maxDance = 0
	maxEnergy = 0


	for song in rows:
		if song[1] > maxValence:
			maxValence = song[1]
		if song[2] > maxTempo:
			maxTempo = song[2]
		if song[3] > maxDance:
			maxDance = song[3]
		if song[4] > maxEnergy:
			maxEnergy = song[4]

	for song in rows:
		cleaned_row = []
		cleaned_row.append(song[1]/maxValence)
		cleaned_row.append(song[2]/maxTempo) 
		data1.append(cleaned_row)
	for song in rows:
		cleaned_row = []
		cleaned_row.append(song[1]/maxValence)
		cleaned_row.append(song[3]/maxDance) 
		data2.append(cleaned_row)
	for song in rows:
		cleaned_row = []
		cleaned_row.append(song[1]/maxValence)
		cleaned_row.append(song[4]/maxEnergy)
		data3.append(cleaned_row)
	for song in rows:
		cleaned_row = []
		cleaned_row.append(song[2]/maxTempo)
		cleaned_row.append(song[3]/maxDance)
		data4.append(cleaned_row)
	for song in rows:
		cleaned_row = []
		cleaned_row.append(song[2]/maxTempo)
		cleaned_row.append(song[4]/maxEnergy)
		data5.append(cleaned_row)
	for song in rows:
		cleaned_row = []
		cleaned_row.append(song[3]/maxDance)
		cleaned_row.append(song[4]/maxEnergy)
		data6.append(cleaned_row)
		

	data1 = np.asarray(data1) #valence and tempo
	data2 = np.asarray(data2) #valence and danceability
	data3 = np.asarray(data3) #valence and energy
	data4 = np.asarray(data4) #tempo and danceability
	data5 = np.asarray(data5) #tempo and energy
	data6 = np.asarray(data6) #danceability and energy



	"""
	variable data is now a 2D numpy array, each row being a list of the song name, valence, tempo, danceability,
	energy.
	"""
	# data_points1 = []
	# data_points2 = []
	# data_points3 = []
	# data_points4 = []
	# data_points5 = []
	# data_points6 = []

	# for i in range(len(data)):
	# 	data_points1.append(np.float_((data[i][1:]))) #valence and tempo
	#     data_points2.append(np.float_((data[i][1:]))) #valence and danceability
	#     data_points3.append(np.float_((data[i][1:]))) #valence and energy
	#     data_points4.append(np.float_((data[i][1:]))) #tempo and danceability
	#     data_points5.append(np.float_((data[i][1:]))) #tempo and energy
	#     data_points6.append(np.float_((data[i][1:]))) #danceability and energy
	# data_points = np.array(data_points)

	clusters = 5
	errors = []

	kms= KMeans(n_clusters=clusters)
	
	errors.append(-1*kms.score(data1)) #score is the error
	errors.append(-1*kms.score(data2))
	errors.append(-1*kms.score(data3))
	errors.append(-1*kms.score(data4))
	errors.append(-1*kms.score(data5))
	errors.append(-1*kms.score(data6))
	
	x = np.arange(6)

	fig, ax = plt.subplots()
	# ax.yaxis.set_major_formatter(formatter)
	plt.bar(x, errors)
	plt.xticks(x, ('V and T', 'V and D', 'V and E', 'T and D', 'T and E', 'D and E'))
	plt.show()

if __name__ == '__main__':
	main()
