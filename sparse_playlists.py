import utils_playlists as utils
from statistics import mean
import numpy as np
import time
import pdb
import pickle
from scipy.spatial.distance import pdist, squareform, cosine
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

"""
README

This program, alongside utils_playlists.py, creates a similarity matrix
using song vectors and context windows.

IMPORTANT: It draws data from the playlist2tracks_full.txt file

Just to clarify, our full_data.json includes 4000 playlists with approximately
not unique 500,000 songs, and playlist2tracks_full is a txt file from our main db.

"""

def plot_word_clusters(data, centroids, centroid_indices):
    """
    DO NOT CHANGE ANYTHING IN THIS FUNCTION

    You can use this function to plot the words and centroids to visualize your code.
    Points with the same color are considered to be in the same cluster.

    :param data - the data set stores as a 2D np array (given in the main function stencil)
    :param centroids - the coordinates that represent the center of the clusters
    :param centroid_indices - the index of the centroid that corresponding data point it closest to

    NOTE: function only works for K <= 6 clusters
    """
    Y = data[:,0]
    x = data[:,1].astype(np.float)
    y = data[:,2].astype(np.float)
    fig, ax = plt.subplots()
    for c in centroids:
        x = np.append(x,c[0])
        y = np.append(y,c[1])
    try:
        colors = {0: 'red', 1: 'yellow', 2: 'blue', 3: 'green', 4: 'brown', 5: 'orange'}
        color = [colors[l] for l in centroid_indices]
        for i in range(len(centroids)):
            color.append('black')
    except KeyError:
        print ("Keep to less than 5 clusters")
        return

    plt.scatter(x,y,c = color)
    plt.show()

#Plots error graph for determining number of clusters in KMeans.
def elbow_point_plot(cluster, errors):
    """
    DO NOT CHANGE ANYTHING IN THIS FUNCTION

    This function helps create a plot representing the tradeoff between the number of clusters
    and the mean squared error.

    :param cluster: 1D np array that represents K (the number of clusters)
    :param errors: 1D np array that represents the mean squared error
    """

    plt.plot(cluster,errors)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Error')
    plt.show()

#Plot TSNE graph
def plot_with_labels(low_dim_embs, labels, filename):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(200, 200))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(
        label,
        xy=(x, y),
        xytext=(5, 2),
        textcoords='offset points',
        ha='right',
        va='bottom')

  plt.savefig(filename)

def vectorization():
    #hyperparameter
    NUM_CLOSEST = 5 #controls the number of closest songs the the song you want
    TRUNC_AMOUNT = 5000 #this controls how many top songs you want to keep in truncation

    """
    I want to make a structure where the corpus is represented by each row
    being a playlist, and then each element in that list to be the songs
    in that list
    """
    corpus = utils.load_corpus()


    #avg length of playlists is the avg length of each row of corpus
    avg_playlist_len = utils.avgLength(corpus)
    print('average playlist length: ' + str(avg_playlist_len))

    #each playlist is truncated to include only the top 5000 most common songs
    trunc_corpus, new_counts = utils.trunc_corpus(corpus, utils.counts(corpus), TRUNC_AMOUNT)

    #average length of playlist is now halved, from ~66 to ~33 songs
    trunc_avg_playlist_len = utils.avgLength(trunc_corpus)
    print('average truncated playlists length: ' + str(trunc_avg_playlist_len))

    # NOTE: Window size parameter depends on the average length of the playlist
    WINDOW_SIZE = int(avg_playlist_len)

    #generate vocab, where each song maps to songid, and vice versa in inverse_vocab from the trucnated playlist data
    vocab, inverse_vocab = utils.construct_vocab(trunc_corpus)

    #generating lookup table
    lookup_table = utils.word_vectors(trunc_corpus, vocab, WINDOW_SIZE)
    lookup_table = np.asarray(lookup_table)
    print("Lookup table:")
    print(lookup_table)

    return lookup_table, vocab, inverse_vocab

#Helper function containing all clustering and playlist generation logic.
def cluster(lookup_table, vocab, inverse_vocab):
    # cosine similarity between each row of look_up table
    distances = squareform(pdist(lookup_table, 'cosine'))
    
    # save distances for future use since it is a time-consuming task
    np.save('distances',distances)

    # load distances if generated already since this takes time for a large dataset
    # distances = np.load('distances.npy')


    tsne = TSNE(
        perplexity=30, n_components=2, init='pca', n_iter=1000, method='exact')

    #dimensionally reduce 5000 features into 2 using TSNE, and saving this file in current directory.
    #Take around 30 minutes for 5000 songs
    low_dim_embs = tsne.fit_transform(distances)

    # # dump low_dim_embs for future use since it is a time-consuming task
    pickle.dump(low_dim_embs, open('low_dim_embs', "wb"))

    labels = [inverse_vocab[i] for i in range(len(distances))]

    # load dimensionally reduced matrix if exits.
    # low_dim_embs = pickle.load(open("low_dim_embs", "rb"))
    # plot_with_labels(low_dim_embs, labels, 'tsne.png')

    # code to determine optimum number of clusters to use, based on error plot result
    K = []
    error = []
    for i in range(1, 10):
        K.append(i)
        kMeans = KMeans(n_clusters = i, max_iter = 50).fit(low_dim_embs)
        centroids = kMeans.cluster_centers_
        indices = kMeans.labels_
        e = -1*kMeans.score(low_dim_embs)
        error.append(e)

    elbow_point_plot(np.array(K), np.array(error))


    # visualizes the clustering results, and generates playlist
    kMeans = KMeans(n_clusters = 500, max_iter = 50).fit(low_dim_embs)
    indices = kMeans.labels_
    data = []
    for index, title in enumerate(labels):
        data.append([title, low_dim_embs[index][0], low_dim_embs[index][1]])
    data = np.array(data)
    centers = kMeans.cluster_centers_
    # graph plot
    plot_word_clusters(data, centers, indices)
    # create dictionary where key is playlist number, and value is array of songs in the playlist
    playlists = {}
    for i in range(0, len(centers)):
        playlists[i] = []
    for  index, value in enumerate(indices):
        playlists[value].append(inverse_vocab[index])

    # playlist generating code
    
    for k in playlists:
        songs = playlists[k]
        for index, song in enumerate(songs):
            # outputs song to playlist file in sub-directory
            print(song[0] + " - " + song[1], file= open('playlists/playlist'+str(k)+'.txt', 'a+'))

def main():
    lookup_table, vocab, inverse_vocab = vectorization()
    cluster(lookup_table, vocab, inverse_vocab)


if __name__ == '__main__':
    main()
