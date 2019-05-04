import json
import numpy as np
from scipy.spatial.distance import cosine
import collections
import pickle

def load_corpus():
    """ Reads the data from disk.
        Returns a list of sentences, where each sentence is split into a list of word tokens
    """
    return pickle.load(open("DATA/FULL DATA/playlist2tracks_full.txt", "rb"))

def make_corpus(playlists):
    #this makes an empty 2d array of length of playlists (4000)
    corpus = [[] for i in range(len(playlists))]

    for i in range(len(playlists)):
        # playlist['tracks'] is a list of dictionaries, each dictionary
        # representing one song. So to access a song name, you do:
        for j in range(len(playlists[i]['tracks'])):
            corpus[i].append(playlists[i]['tracks'][j]['track_name'])

    #now corpus is a 2d array, each row represents a playlist, each 
    #element in a row is the name of a song in that playlist

    return corpus

def counts(corpus):
    """ Given a corpus (such as returned by load_corpus), return a dictionary
        of word frequencies. Maps string token to integer count.
    """
    return collections.Counter(w for s in corpus for w in s)

def trunc_corpus(corpus, counts, truncated_amount):
    """ Limit the vocabulary to the 5k most-frequent words. Remove rare words from
         the original corpus.
        Input: A list of list of string. Each string represents a word token.
        Output: A tuple (new_corpus, new_counts)
                new_corpus: A corpus (list of list of string) with only the 5k most-frequent words
                new_counts: Counts of the 5k most-frequent words

        Hint: Sort the keys of counts by their values
    """
    # can use Counter.most_common()
    new_counts = dict(counts.most_common(truncated_amount))
    new_corpus = []
    for playlist in corpus:
        new_playlist = []
        for song, artist in playlist:
            if (song, artist) in new_counts.keys():
                new_playlist.append( (song, artist) )
        new_corpus.append(new_playlist)
    return (new_corpus, new_counts)

def avgLength(lst):
    lengths = [len(i) for i in lst]
    return 0 if len(lengths) == 0 else (float(sum(lengths))/len(lengths))

def construct_vocab(corpus):
    """
        Input: A list of list of string. Each string represents a word token.
        Output: A tuple of dicts: (vocab, inverse_vocab)
                vocab: A dict mapping str -> int. This will be your vocabulary.
                inverse_vocab: Inverse mapping int -> str
        each unique word is given an id
    """
    # raise NotImplementedError("construct_vocab")
    vocab = {}
    inverse_vocab = {}
    counter = 0
    for playlist in corpus:
        for tup in playlist:
            if tup not in vocab:
                vocab[tup] = counter
                inverse_vocab[counter] = tup
                counter += 1
    return (vocab, inverse_vocab)

def word_vectors(corpus, vocab, WINDOW_SIZE):
    """
        Input: A corpus (list of list of string) and a vocab (word-to-id mapping)
        Output: A lookup table that maps [word id] -> [word vector]
    """
    """
    The word vector of word w would be a list of the size of the vocabulary where the 
    number at each index indicates the number of times a word (with that ID represented 
    at the index) has appeared in the context window of w. Thus, repeated words in the 
    window of w would increment the number at that specific index.

    the context window should not span across diff playlists.
    """
    #here I've made a lookup table with a row for each word id, and with value that is a
    #np array of the length of vocab, filled with zeros for now.
    lookup_table = [np.zeros(len(vocab)) for i in range(len(vocab))]
    # next, for each sentence in the corpus, I go through and fill in the corresponding
    # word vectors for each word in the sentence in relation to it's context window
    context_window = [i for i in range(-WINDOW_SIZE, WINDOW_SIZE +1) if i != 0] #

    for i in range(len(corpus)):
        for j in range(len(corpus[i])):
            curr_tup = corpus[i][j]
            curr_id = vocab[curr_tup]
            for index in context_window:
                #this is to keep the bounds of the sentence
                if (j+index >= 0) and (j+index < len(corpus[i])):
                    tup = corpus[i][j+index]
                    tupID = vocab[tup]
                    lookup_table[curr_id][tupID] += 1
    
    return lookup_table





