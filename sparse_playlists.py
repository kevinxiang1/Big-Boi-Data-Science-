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

def plot_word_clusters(data, centroids, centroid_indices):
    """
    DO NOT CHANGE ANYTHING IN THIS FUNCTION

    You can use this function to plot the words and centroids to visualize your code.
    Points with the same color are considered to be in the same cluster.

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
        colors = {0: 'red', 1: 'yellow', 2: 'blue', 3: 'green', 4: 'brown', 5: 'orange'}
        color = [colors[l] for l in centroid_indices]
        for i in range(len(centroids)):
            color.append('black')
    except KeyError:
        print ("Keep to less than 5 clusters")
        return

    plt.scatter(x,y,c = color)
    # plt.xlabel('Neutral --> Polarizing')
    # plt.ylabel('Negative --> Positive')
    plt.show()




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

# def get_closest_songs(distances, inverse_vocab):
#     distances = np.array(distances)
#     sort = np.argsort(distances)[1:NUM_CLOSEST+1]
#     closest = []
#     for wid in sort:
#         closest.append(inverse_vocab[wid])
#     print(sort)
#     return sort, closest

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

def main():
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
    # trunc_corpus, new_counts = utils.trunc_corpus(corpus, utils.counts(corpus), TRUNC_AMOUNT)

    #average length of playlist is now halved, from ~66 to ~33 songs
    # trunc_avg_playlist_len = utils.avgLength(trunc_corpus)
    # print('average truncated playlists length: ' + str(trunc_avg_playlist_len))

    # NOTE: Window size parameter depends on the average length of the playlist
    WINDOW_SIZE = int(avg_playlist_len)

    vocab, inverse_vocab = utils.construct_vocab(corpus)

    #generating lookup table
    lookup_table = utils.word_vectors(corpus, vocab, WINDOW_SIZE)
    lookup_table = np.asarray(lookup_table)
    print("Lookup table:")
    print(lookup_table)

    clustering(lookup_table, vocab, inverse_vocab)


def clustering(lookup_table, vocab, inverse_vocab):

    distances = squareform(pdist(lookup_table, 'cosine'))
    print(distances)
    np.save('distances',distances)

    # distances = np.load('distances.npy')

    # most_sim = (None, None)
    # most_val = float("inf")
    # least_sim = (None, None)
    # least_val = float("-inf")
    # for songid, row in enumerate(distances):
    #     for nid, dist in enumerate(row):
    #         if songid!= nid:
    #             if dist < most_val:
    #                 most_val = dist
    #                 most_sim = (inverse_vocab[songid], inverse_vocab[nid])
    #             if dist > least_val:
    #                 least_val = dist
    #                 least_sim = (inverse_vocab[songid], inverse_vocab[nid])

    # print(most_sim)
    # print(most_val)
    # print(least_sim)
    # print(least_val)
    # sort, closest = get_closest_songs(distances[vocab["HUMBLE."]], inverse_vocab)
    # print(closest)


    tsne = TSNE(
        perplexity=30, n_components=2, init='pca', n_iter=1000, method='exact')

    # plot_only = 5000
    # data_set = distances[:plot_only, :]

    low_dim_embs = tsne.fit_transform(distances)
    print(low_dim_embs)
    pickle.dump(low_dim_embs, open('low_dim_embs', "wb"))

    labels = [inverse_vocab[i] for i in range(len(distances))]

    # low_dim_embs = pickle.load(open("low_dim_embs", "rb"))

    #plot_with_labels(low_dim_embs, labels, 'tsne.png')

    K = []
    error = []

    # for i in range(1, 10):
    #     K.append(i)
    #     kMeans = KMeans(n_clusters = i, max_iter = 50).fit(low_dim_embs)
    #     centroids = kMeans.cluster_centers_
    #     indices = kMeans.labels_
    #     # e = -1*kMeans.score(low_dim_embs)
    #     # error.append(e)

    # elbow_point_plot(np.array(K), np.array(error))

    kMeans = KMeans(n_clusters = 6, max_iter = 50).fit(low_dim_embs)
    indices = kMeans.labels_
    data = []
    for index, title in enumerate(labels):
        data.append([title, low_dim_embs[index][0], low_dim_embs[index][1]])
    data = np.array(data)
    centers = kMeans.cluster_centers_
    plot_word_clusters(data, centers, indices)
    playlists = {}
    for i in range(0, len(centers)):
        playlists[i] = []
    for  index, value in enumerate(indices):
        playlists[value].append(inverse_vocab[index])

    for k in playlists:
        songs = playlists[k]
        for index, song in enumerate(songs):
            print(song, file= open('playlists/playlist'+str(k)+'.txt', 'a+'))
    print(len(distances))


if __name__ == '__main__':
    main()
