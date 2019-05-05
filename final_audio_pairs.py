from skimage import io
import numpy as np
import random
import numpy.matlib
import math
import csv
import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import ast
import sqlite3
import hypertools as hyp
import pickle

#import kmeans template

def sk_learn_cluster(X, K):
	"""
	:param X: 2D np array containing audio features
	:param K: number of clusters
	:return: predictions and cluster centers
	"""
	kms = KMeans(K)
	kms.fit(X)
	return (kms.cluster_centers_, kms.predict(X))


def plot_word_clusters(data, centroids, centroid_indices, x_axis, y_axis):
	"""
	:param data - the data set stores as a 2D np array (given in the main function stencil)
	:param centroids - the coordinates that represent the center of the clusters
	:param centroid_indices - the index of the centroid that corresponding data point it closest to

	NOTE: function only works for K <= 5 clusters
	"""
	Y = data[:,0]
	x = data[:,1].astype(np.float)
	y = data[:,2].astype(np.float)
	fig, ax = plt.subplots()
	for c in centroids:
		x = np.append(x,c[0])
		y = np.append(y,c[1])
	try:
		colors = {0: 'red', 1: 'yellow', 2: 'blue', 3: 'green', 4: 'brown'}
		color = [colors[l] for l in centroid_indices]
		for i in range(len(centroids)):
			color.append('black')
	except KeyError:
		print ("Keep to less than 5 clusters")
		return
	#this section of code below adds the song name text as an annotation to the data points
	# for i, txt in enumerate(Y):
	# 	ax.annotate(txt, (x[i], y[i]))
	plt.scatter(x,y,c = color)
	plt.xlabel(x_axis)
	plt.ylabel(y_axis)
	plt.suptitle("SciKit-Learn KMeans Clustering: 5000 songs")
	plt.show()

def elbow_point_plot(clusters, errors):
	"""
	This function helps create a plot representing the tradeoff between the number of clusters
	and the mean squared error.

	:param cluster: 1D np array that represents K (the number of clusters)
	:param errors: 1D np array that represents the mean squared error

	WHEN THIS PRINTS, RESCALE THE WINDOW TO SHOW THE AXES
	"""
	fig = plt.plot(clusters,errors)
	plt.xlabel('Number of Clusters')
	plt.ylabel('Error')
	plt.suptitle("Elbow Point Plot: 200 playlists")
	plt.show()

def main():
	"""
	This function loads the data set as a 2D numpy array in the data variable
	"""

	# The following codes loads the data set into a 2D np array called data
	# with open('data/word_sentiment.csv') as words_file:

	# file = open("./previous data/500_songs.txt", "r")
	# conn = sqlite3.connect('tracks2features_full.db')
	# c = conn.cursor()
	# c.execute("SELECT * FROM audio_features")
	# rows = c.fetchall()
	with open('./DATA/FULL DATA TRUNCATED/tracks2features_full_truncated.csv') as f:
		reader = csv.reader(f)
		rows = list(reader)
		rows = rows[1:]

	maxValence = 0
	maxTempo = 0
	maxDance = 0
	maxEnergy = 0
	# maxSpeech = 0
	new_rows = []
	for row in rows:
		x = [float(feature) for feature in row[1:]] 
		# print("x is" + str(x))
		new_rows.append([row[0]]+ x)
	# print(type(new_rows[0][3]))
	for song in new_rows:
		if song[1] > maxValence:
			maxValence = song[1]
		if song[2] > maxTempo:
			maxTempo = song[2]
		if song[3] > maxDance:
			maxDance = song[3]
		if song[4] > maxEnergy:
			maxEnergy = song[4]
		# if song[5] > maxSpeech:
		# 	maxSpeech = song[5]
	
	data1 = []
	data2 = []
	data3 = []
	data4 = []
	data5 = []
	data6 = []

	fulldata1 = []
	fulldata2 = []
	fulldata3 = []
	fulldata4 = []
	fulldata5 = []
	fulldata6 = []

	#calculates maximum values of each audio feature in our dataset
	for song in new_rows:
		if song[1] > maxValence:
			maxValence = song[1]
		if song[2] > maxTempo:
			maxTempo = song[2]
		if song[3] > maxDance:
			maxDance = song[3]
		if song[4] > maxEnergy:
			maxEnergy = song[4]
			
	#creates arrays of the combinations of audio feature pairs for each song
	for song in new_rows:
		cleaned_row = []
		cleaned_row.append(song[0])
		cleaned_row.append(song[1]/maxValence)
		cleaned_row.append(song[2]/maxTempo) 
		fulldata1.append(cleaned_row)
		super_clean_row = cleaned_row[1:]
		data1.append(super_clean_row)
	

	for song in new_rows:
		cleaned_row = []
		cleaned_row.append(song[0])
		cleaned_row.append(song[1]/maxValence)
		cleaned_row.append(song[3]/maxDance) 
		fulldata2.append(cleaned_row)
		super_clean_row = cleaned_row[1:]
		data2.append(super_clean_row)
	for song in new_rows:
		cleaned_row = []
		cleaned_row.append(song[0])
		cleaned_row.append(song[1]/maxValence)
		cleaned_row.append(song[4]/maxEnergy)
		fulldata3.append(cleaned_row)
		super_clean_row = cleaned_row[1:]
		data3.append(super_clean_row)
	for song in new_rows:
		cleaned_row = []
		cleaned_row.append(song[0])
		cleaned_row.append(song[2]/maxTempo)
		cleaned_row.append(song[3]/maxDance)
		fulldata4.append(cleaned_row)
		super_clean_row = cleaned_row[1:]
		data4.append(super_clean_row)
	for song in new_rows:
		cleaned_row = []
		cleaned_row.append(song[0])
		cleaned_row.append(song[2]/maxTempo)
		cleaned_row.append(song[4]/maxEnergy)
		fulldata5.append(cleaned_row)
		super_clean_row = cleaned_row[1:]
		data5.append(super_clean_row)
	for song in new_rows:
		cleaned_row = []
		cleaned_row.append(song[0])
		cleaned_row.append(song[3]/maxDance)
		cleaned_row.append(song[4]/maxEnergy)
		fulldata6.append(cleaned_row)
		super_clean_row = cleaned_row[1:]
		data6.append(super_clean_row)
	
	#turns above arrays into numpy arrays
	data1 = np.asarray(data1) #valence and tempo
	data2 = np.asarray(data2) #valence and danceability
	data3 = np.asarray(data3) #valence and energy
	data4 = np.asarray(data4) #tempo and danceability
	data5 = np.asarray(data5) #tempo and energy
	data6 = np.asarray(data6) #danceability and energy

	fulldata1 = np.asarray(fulldata1) #valence and tempo
	fulldata2 = np.asarray(fulldata2) #valence and danceability
	fulldata3 = np.asarray(fulldata3) #valence and energy
	fulldata4 = np.asarray(fulldata4) #tempo and danceability
	fulldata5 = np.asarray(fulldata5) #tempo and energy
	fulldata6 = np.asarray(fulldata6) #danceability and energy

	sklearn_kms1 = sk_learn_cluster(data1, 5)
	plot_word_clusters(fulldata1, sklearn_kms1[0], sklearn_kms1[1], 'Valence', 'Tempo')

	sklearn_kms2 = sk_learn_cluster(data2, 5)
	plot_word_clusters(fulldata2, sklearn_kms2[0], sklearn_kms2[1], 'Valence', 'Danceability')

	sklearn_kms3 = sk_learn_cluster(data3, 5)
	plot_word_clusters(fulldata3, sklearn_kms3[0], sklearn_kms3[1], 'Valence', 'Energy')

	sklearn_kms4 = sk_learn_cluster(data4, 5)
	plot_word_clusters(fulldata4, sklearn_kms4[0], sklearn_kms4[1], 'Tempo', 'Danceability')

	sklearn_kms5 = sk_learn_cluster(data5, 5)
	plot_word_clusters(fulldata5, sklearn_kms5[0], sklearn_kms5[1], 'Tempo', 'Energy')

	sklearn_kms6 = sk_learn_cluster(data6, 5)
	plot_word_clusters(fulldata6, sklearn_kms6[0], sklearn_kms6[1], 'Energy', 'Danceability')




if __name__ == '__main__':
	main()