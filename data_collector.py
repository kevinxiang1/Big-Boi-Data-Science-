#!/usr/bin/python
# -*- coding: utf-8 -*-

## main imports
import os
import urllib
import re
import json
import csv
import requests
import sqlite3
from datetime import timedelta, date
from time import sleep
from bs4 import BeautifulSoup
import time
import pickle

## spotify API imports
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

"""
    Load the playlist data and initialize the Spotipy object for API calls.
    returns:
        raw: the raw playlist data extracted from the JSON document
        sp: the spotipy object that we will use for API calls
"""
def __init__():
    # load JSON at index playlists
    with open('full_data.json') as f:
        raw = json.load(f)['playlists']

    # Spotify API objects instantiation
    client_credentials_manager = SpotifyClientCredentials(client_id='2fd6c37e0dc44ebbbf7275b7a24ce182', client_secret='57b21bdf2a9c4a888a5a75661c1804a2')
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    return raw, sp

"""
    Construct the main corpus of the data (a table in which each row
    is the features of a playlist). the 7th element in each row is
    a list of lists containing the features of songs within each 
    playlist.
    returns:
        data: the main table that we will be using to extract relevant
              features about songs and playlists
"""
def construct_main_corpus(raw):
    data = []
    for i in range(len(raw)): 
        cur_playlist_features = raw[i]
        data.append(list(cur_playlist_features.values()))
        tracks_features = list(data[i])[7]
        data[i][7] = []
        for j, track_features in enumerate(tracks_features):
            data[i][7].append(list(track_features.values()))
    return data

"""
    "DL related data extraction"
    Create a mapping from playlists (sentences) to songs that comprise them
    (words), which will be later used to create a model that identifies the
    closest songs to a given song using similar techniques to those that we
    used in the DL assignment.
    returns:
        corpus: the mapping from playlist titles to the titles of track within it:
                index i in corpus is representative of the i-th playlist and 
                at each index, we store a tuple of the format (p, [t1, t2, ..., tn]):
                       - p is the playlist title
                       - tj is the title of the j-th track in p
"""
def construct_playlist_to_tracks(data):
    corpus = [[] for i in range(len(data))]
    for i, playlist in enumerate(data):
        tracks = []
        for j, track in enumerate(playlist[7]):
            cur_track_title, cur_track_artist = track[4], track[1]
            tracks.append( (cur_track_title, cur_track_artist) )
        corpus[i] = tracks
    return corpus

"""
    "Clustering (KMeans) related data extraction"
    Create a mapping from songs to a select set of audio features that we want
    to use to compare them. This is done via API calls to an official Spotify
    website. We will be using these features to compare songs and cluster them
    using KMeans.
    returns:
        tracks2features: the mapping from track titles to its audio features (valence,
                         tempo, danceability, energy, speechiness). This is a dictionary
                         that maps a track title to a list of its audio features.
"""
def construct_song_to_features(sp, data):
    tracks2features = dict()
    for i, playlist in enumerate(data):
        playlist_title = playlist[0]
        for j, track in enumerate(playlist[7]):
            cur_track_title, cur_track_uri = track[4], track[2]
            if cur_track_title not in tracks2features:
                audio_fts = sp.audio_features(cur_track_uri)[0]
                relevant_features = [float(audio_fts['valence']), float(audio_fts['tempo']), float(audio_fts['danceability']), float(audio_fts['energy']), float(audio_fts['speechiness'])]
                tracks2features[cur_track_title] = relevant_features
    return tracks2features

"""
    Create a database containing songs and their audio features. We used the techniques
    we learned in the scraping assignment to construct this database.
"""
def save_data(tracks2features):
    # Create connection to database
    conn = sqlite3.connect('tracks2features.db')
    c = conn.cursor()

    # Delete tables if they exist
    c.execute('DROP TABLE IF EXISTS "audio_features";')

    c.execute('''CREATE TABLE audio_features(track_title TEXT,
                    valence FLOAT, tempo FLOAT, danceability FLOAT,
                    energy FLOAT, speechiness FLOAT, PRIMARY KEY(track_title));''')

    for song in tracks2features:
        c.execute('INSERT INTO audio_features VALUES (?, ?, ?, ?, ?, ?)', (
            song,
            tracks2features[song][0],
            tracks2features[song][1],
            tracks2features[song][2],
            tracks2features[song][3],
            tracks2features[song][4]
            ))
    conn.commit()

def main():
    start = time.time()
    raw_data, sp = __init__()
    main_table = construct_main_corpus(raw_data)
    playlist2tracks = construct_playlist_to_tracks(main_table)
    # tracks2features = construct_song_to_features(sp, main_table)
    # save_data(tracks2features)
    with open("playlist2tracks.txt", "wb") as fp:
        pickle.dump(playlist2tracks, fp)
    end = time.time()
    print("Program ran for: " + str(end-start) + " seconds")

main()