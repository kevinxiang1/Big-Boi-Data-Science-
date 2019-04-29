#!/usr/bin/python
# -*- coding: utf-8 -*-

## main imports
import os
import urllib
import re
import json
import requests
import sqlite3
from datetime import timedelta, date
from time import sleep
from bs4 import BeautifulSoup

## spotify API imports
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

def __init__():
    # load JSON at index playlists
    with open('full_data.json') as f:
        raw = json.load(f)['playlists']

    # Spotify API objects instantiation
    client_credentials_manager = SpotifyClientCredentials(client_id='2fd6c37e0dc44ebbbf7275b7a24ce182', client_secret='57b21bdf2a9c4a888a5a75661c1804a2')
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    return raw, sp

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


# DL related data extraction
    # corpus:
    # index i is representative of the i-th playlist
    # at each index, I store a tuple of the format (p, [t1, t2, ..., tn]) where
    #       p is the playlist title
    #       ti is the title of the i-th track in p
    # sentences are playlist names
    # words are track names
    # each sentence (playlist) is mapped to the set of words that comprise it (tracks it includes)
    #
    # if-clause at the end helps in building a dicionary from track_title -> playlists including the track
def construct_playlist_to_tracks(data):
    corpus = [[] for i in range(len(data))]
    for i, playlist in enumerate(data):
        playlist_title = playlist[0]
        track_titles = []
        for j, track in enumerate(playlist[7]):
            cur_track_title, cur_track_uri = track[4], track[2]
            track_titles.append(cur_track_title)
        corpus[i] = (playlist_title, track_titles)
    return corpus

# Clustering (KMeans and KNN) related data extraction
def construct_song_to_features(sp, data, howmuch):
    songs2features = dict()
    for i, playlist in enumerate(data[:howmuch]):
        playlist_title = playlist[0]
        for j, track in enumerate(playlist[7]):
            cur_track_title, cur_track_uri = track[4], track[2]
            if cur_track_title not in songs2features:
                audio_features = sp.audio_features(cur_track_uri)[0]
                relevant_features = [float(audio_features['valence']), float(audio_features['tempo']), float(audio_features['danceability']), float(audio_features['energy']), float(audio_features['speechiness'])]
                songs2features[cur_track_title] = (relevant_features, {playlist_title})
            else:
                songs2features[cur_track_title][1].add(playlist_title)
    return songs2features

# Classification related data extraction
    # TODO
def collect_song_labels(data):
    pass

def create_database(data, data_name):
    # Create connection to database
    conn = sqlite3.connect(data_name)
    c = conn.cursor()

    # Delete tables if they exist
    c.execute('DROP TABLE IF EXISTS "symbols";')
    c.execute('DROP TABLE IF EXISTS "quotes";')

    c.execute('''CREATE TABLE symbols(symbol TEXT,
                    name TEXT, PRIMARY KEY(symbol));''')
    c.execute('''CREATE TABLE quotes(
                    symbol TEXT, price float, avg_price float, num_articles int,
                    market_cap float, change float, PRIMARY KEY(symbol));''')

    for i in range(len(symbols)):
        c.execute('INSERT INTO symbols VALUES (?, ?)', (symbols[i],
                company_names[i]))
        c.execute('INSERT INTO quotes VALUES (?, ?, ?, ?, ?, ?)', (
            symbols[i],
            prices[i],
            market_caps[i],
            daily_percentage_change[i],
            avg_closing_price_6m[i],
            recent_articles_count[i],
            ))
    conn.commit()

def main():
    raw_data, sp = __init__()
    main_table = construct_main_corpus(raw_data)
    playlist2tracks = construct_playlist_to_tracks(main_table)
    songs2features = construct_song_to_features(sp, main_table, 500)
    print(songs2features)
    
main()