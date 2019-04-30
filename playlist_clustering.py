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

NUM_CLOSEST = 50
def main():
    lookup_table = np.load('lookup_table.npy')
    print(lookup_table)
    print(len(lookup_table[1]))
    playlist = np.load('playlist.npy') 
    print(len(playlist))

    # distances = squareform(pdist(lookup_table, 'cosine'))
    # np.save('distances',distances)


    distances = np.load('distances.npy')
    inverse_vocab = pickle.load(open("inverse_vocab", "rb"))
    vocab = pickle.load(open("vocab", "rb"))
    print(distances)
    most_sim = (None, None)
    most_val = float("inf")
    least_sim = (None, None)
    least_val = float("-inf")
    for songid, row in enumerate(distances):
        for nid, dist in enumerate(row):
            if songid!= nid:
                if dist < most_val:
                    most_val = dist
                    most_sim = (inverse_vocab[songid], inverse_vocab[nid])
                if dist > least_val:
                    least_val = dist
                    least_sim = (inverse_vocab[songid], inverse_vocab[nid])

    print(most_sim)
    print(most_val)
    print(least_sim)
    print(least_val)
    sort, closest = get_closest_songs(distances[vocab["HUMBLE."]], inverse_vocab)
    print(closest)


    tsne = TSNE(
        perplexity=30, n_components=2, init='pca', n_iter=1000, method='exact')

    # new_set = distances[vocab["HUMBLE."]]
    # labels = ["HUMBLE."]
    # print(new_set)
    # for index in sort:
    #     new_set = np.concatenate((new_set, distances[index]), axis = 0)
    #     labels.append(inverse_vocab[index])
    # new_set = new_set.reshape((NUM_CLOSEST+1, len(distances[0])))
    # print(new_set.shape)
    # low_dim_embs = tsne.fit_transform(new_set)
    # plot_with_labels(low_dim_embs, labels, 'tsne.png')


    plot_only = 5000
    data_set = distances[:plot_only, :]
    print(data_set)
    # low_dim_embs = tsne.fit_transform(data_set)
    # pickle.dump(low_dim_embs, open('low_dim_embs', "wb"))
    # print("finished")
    labels = [inverse_vocab[i] for i in range(plot_only)]

    # plot_with_labels(low_dim_embs, labels, 'tsne.png')

    low_dim_embs = pickle.load(open("low_dim_embs", "rb"))
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

    kMeans = KMeans(n_clusters = 500, max_iter = 50).fit(low_dim_embs)
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
        print(playlists[k], file= open('playlists/playlist'+str(k)+'.txt', 'w'))



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

def get_closest_songs(distances, inverse_vocab):
    distances = np.array(distances)
    sort = np.argsort(distances)[1:NUM_CLOSEST+1]
    closest = []
    for wid in sort:
        closest.append(inverse_vocab[wid])
    print(sort)
    return sort, closest

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


if __name__ == '__main__':
    main()

