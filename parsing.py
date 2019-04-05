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

def main():

    data = []

    # load JSON at index playlists
    with open('full_data.json') as f:
        raw = json.load(f)['playlists']

    # Spotify API objects instantiation
    client_credentials_manager = SpotifyClientCredentials(client_id='2fd6c37e0dc44ebbbf7275b7a24ce182', client_secret='57b21bdf2a9c4a888a5a75661c1804a2')
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    songs = dict()

    
    for i in range(len(raw)):

        cur_playlist = raw[i]
        data.append(cur_playlist.values())

        tracks = data[i][7]
        data[i][7] = []
        playlist_name = data[i][0]

        for j, track in enumerate(tracks):
            data[i][7].append(track.values())

            track_name = data[i][7][j][4]
            track_uri = str(data[i][7][j][2])

            if track_name not in songs:
                audio_features = [sp.audio_features(track_uri)[0]['valence'], sp.audio_features(track_uri)[0]['tempo']]
                songs[track_name] = (audio_features, [playlist_name])
            else:
                songs[track_name][1].append(playlist_name)

        
    for song in songs:
        print("Title:")
        print(song)
        print("Valence, Tempo:")
        print(songs[song][0])
        print("Playlists where it appears:")
        print(songs[song][1])
        print("---------")

    # at this point, data is an array of 4000 playlists * 11 attributes
    # the 7th attribute/index is a list of songs. Each song is a list of
    # its attributes



main()