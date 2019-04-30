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




def plot_word_clusters(data, centroids, centroid_indices):
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
	plt.xlabel('Danceability: Undanceable --> Danceable')
	plt.ylabel('Speechiness: Low --> High')
	plt.suptitle("SciKit-Learn KMeans Clustering: 200 Playlists")
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
	conn = sqlite3.connect('tracks2features.db')
	c = conn.cursor()
	c.execute("SELECT * FROM audio_features")
	rows = c.fetchall()

	# my_dict = ast.literal_eval(file.read())
	data = []
	maxValence = 0
	maxTempo = 0
	maxDance = 0
	maxEnergy = 0
	maxSpeech = 0
	for song in rows:
		if song[1] > maxValence:
			maxValence = song[1]
		if song[2] > maxTempo:
			maxTempo = song[2]
		if song[3] > maxDance:
			maxDance = song[3]
		if song[4] > maxEnergy:
			maxEnergy = song[4]
		if song[5] > maxSpeech:
			maxSpeech = song[5]
	for song in rows:
		cleaned_row = []
		cleaned_row.append(song[0])
		#only want to input the valence and tempo values, which is a tuple in the first element of the values
		# cleaned_row.append(song[1]/maxValence)
		# cleaned_row.append(song[2]/maxTempo) 
		cleaned_row.append(song[3]/maxDance)
		# cleaned_row.append(song[4]/maxEnergy)
		cleaned_row.append(song[5]/maxSpeech)
		data.append(cleaned_row)
	data = np.asarray(data)
	# print(data)
	"""
	variable data is now a 2D numpy array, each row being a list of the song name, valence, tempo, danceability,
	energy, and speechiness.

	I want to keep it in this format for plot_word_clusters, but for sklearn, I only need the valence 
	and tempo
	"""
	data_points = []
	for i in range(len(data)):
		data_points.append(np.float_((data[i][1:])))
	data_points = np.array(data_points)

	clusters = np.array([3,4,5,6,7,8,9,10,11])
	errors = []

	for item in clusters:
		kms= KMeans(n_clusters=item)
		kmeans_fit = kms.fit(data_points).predict(data_points)
		kmeans_fit = np.reshape(kmeans_fit, (-1, 1))
		# final_clusters = np.concatenate([data_points, kmeans_fit], axis=1)
		# hyp.plot(final_clusters, '.', n_clusters=item)
		errors.append(-1*kms.score(data_points)) #score is the error
	errors = np.asarray(errors)
	elbow_point_plot(clusters,errors)

	sklearn_kms = sk_learn_cluster(data_points, 5)
	plot_word_clusters(data, sklearn_kms[0], sklearn_kms[1])


if __name__ == '__main__':
	main()
