import utils_playlists as utils
from statistics import mean
import numpy as np
import time
import pdb
import pickle
from scipy.spatial.distance import pdist, squareform, cosine
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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


    plot_only = 2000
    low_dim_embs = tsne.fit_transform(lookup_table[:plot_only, :])
    pickle.dump(low_dim_embs, open('low_dim_embs', "wb"))
    labels = [inverse_vocab[i] for i in range(plot_only)]
    plot_with_labels(low_dim_embs, labels, 'tsne.png')


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
  plt.figure(figsize=(1000, 1000))  # in inches
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

