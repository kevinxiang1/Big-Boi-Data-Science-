from skimage import io
import numpy as np
import random
import numpy.matlib
import math
import csv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import ast
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
	plt.xlabel('Valence: Negative --> Positive')
	plt.ylabel('Tempo(bpm): Slow --> Fast')
	plt.suptitle("SciKit-Learn KMeans Clustering: 500 Songs")
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
	plt.suptitle("Elbow Point Plot: 500 Songs")
	plt.show()

def main():
	"""
	This function loads the data set as a 2D numpy array in the data variable
	"""

	# The following codes loads the data set into a 2D np array called data
	# with open('data/word_sentiment.csv') as words_file:

	file = open("500_songs.txt", "r")
	my_dict = ast.literal_eval(file.read())
	data = []
	print(len(my_dict.keys())) #this prints the number of songs
	for key in my_dict.keys():
		cleaned_row = []
		cleaned_row.append(key)
		#only want to input the valence and tempo values, which is a tuple in the first element of the values
		cleaned_row.append(my_dict[key][0][0])
		cleaned_row.append(my_dict[key][0][1]/200) #divide by 200 to normalize the tempo
		data.append(np.array(cleaned_row))
	data = np.array(data)
	# print(data)
	"""
	variable data is now a 2D numpy array, each row being a list of the song name, valence, and tempo
	I want to keep it in this format for plot_word_clusters, but for sklearn, I only need the valence 
	and tempo
	"""
	data_points = []
	for i in range(len(data)):
		data_points.append(np.float_(data[i][1:]))
	dat = np.asarray(data_points)
	# print(data_points)

	clusters = np.array([1,2,3,4,5,5,6,7,8])
	errors = []
	for item in clusters:
		kms= KMeans(item)
		kms.fit_predict(data_points)
		errors.append(-1*kms.score(data_points)) #score is the error
	errors = np.asarray(errors)
	# elbow_point_plot(clusters,errors)

	sklearn_kms = sk_learn_cluster(data_points, 5)
	plot_word_clusters(data, sklearn_kms[0], sklearn_kms[1])

"""
NOTE, I removed from "100_playlists.txt" and "songs.txt" this outlier:
"u'Super Mario Bros - Original': ([0, 0], [u'PlayStation'])"
because it has 0 tempo and 0 valence
"""


if __name__ == '__main__':
	main()
