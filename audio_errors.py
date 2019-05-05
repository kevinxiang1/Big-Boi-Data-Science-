from skimage import io
import numpy as np
import random
import numpy.matlib
import math
import csv
import matplotlib
matplotlib.use('TkAgg')
from sklearn.cluster import KMeans
import ast
import sqlite3
import hypertools as hyp
import pickle

#import kmeans template

plt = matplotlib.pyplot

def main():
	"""
	This function loads the data set as a 2D numpy array in the data variable
	"""

	# The following codes loads the data set into 2D np arrays

	with open('./DATA/FULL DATA TRUNCATED/tracks2features_full_truncated.csv') as f:
		reader = csv.reader(f)
		rows = list(reader)
		rows = rows[1:]

	new_rows = []
	for row in rows:
		x = [float(feature) for feature in row[1:]] 
		new_rows.append([row[0]]+ x)

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
		cleaned_row.append(song[1]/maxValence)
		cleaned_row.append(song[2]/maxTempo) 
		data1.append(cleaned_row)
	for song in new_rows:
		cleaned_row = []
		cleaned_row.append(song[1]/maxValence)
		cleaned_row.append(song[3]/maxDance) 
		data2.append(cleaned_row)
	for song in new_rows:
		cleaned_row = []
		cleaned_row.append(song[1]/maxValence)
		cleaned_row.append(song[4]/maxEnergy)
		data3.append(cleaned_row)
	for song in new_rows:
		cleaned_row = []
		cleaned_row.append(song[2]/maxTempo)
		cleaned_row.append(song[3]/maxDance)
		data4.append(cleaned_row)
	for song in new_rows:
		cleaned_row = []
		cleaned_row.append(song[2]/maxTempo)
		cleaned_row.append(song[4]/maxEnergy)
		data5.append(cleaned_row)
	for song in new_rows:
		cleaned_row = []
		cleaned_row.append(song[3]/maxDance)
		cleaned_row.append(song[4]/maxEnergy)
		data6.append(cleaned_row)
		
	#turns above arrays into numpy arrays
	data1 = np.asarray(data1) #valence and tempo
	data2 = np.asarray(data2) #valence and danceability
	data3 = np.asarray(data3) #valence and energy
	data4 = np.asarray(data4) #tempo and danceability
	data5 = np.asarray(data5) #tempo and energy
	data6 = np.asarray(data6) #danceability and energy

	clusters = 5
	errors = []

	kms= KMeans(n_clusters=clusters)
	
	#run kmeans clustering on audio feature pairs
	kms.fit_predict(data1)
	kms.fit_predict(data2)
	kms.fit_predict(data3)
	kms.fit_predict(data4)
	kms.fit_predict(data5)
	kms.fit_predict(data6)

	errors.append(-1*kms.score(data1)) #score is the error
	errors.append(-1*kms.score(data2))
	errors.append(-1*kms.score(data3))
	errors.append(-1*kms.score(data4))
	errors.append(-1*kms.score(data5))
	errors.append(-1*kms.score(data6))
	
	x = np.arange(6)

	#creates bar chart
	fig, ax = plt.subplots()
	plt.bar(x, errors)
	plt.ylabel('Error')
	plt.title('Clustering Errors for Audio Feature Pairs')
	plt.xticks(x, ('V and T', 'V and D', 'V and E', 'T and D', 'T and E', 'D and E'))
	plt.show()

if __name__ == '__main__':
	main()
