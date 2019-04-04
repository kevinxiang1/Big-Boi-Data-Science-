#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import urllib
import re
import json
import requests
import sqlite3
from datetime import timedelta, date
from time import sleep
from bs4 import BeautifulSoup

def main():
    data = []
    with open('full_data.json') as f:
        raw = json.load(f)['playlists']

    
    for i in range(len(raw)):

        cur_playlist = raw[i]
        data.append(cur_playlist.values())

        tracks = data[i][7]
        data[i][7] = []
        for track in tracks:
            data[i][7].append(track.values())
    print(data)
    return(data)
    # for i in range(len(data[0])):
    #     if i !=7: 
    #         print(data[1][i])
    #     else:
    #         for song in data[1][i]:
    #             print(song)

    # at this point, data is an array of 4000 playlists * 11 attributes
    # the 7th attribute/index is a list of songs. Each song is a list of
    # its attributes

main()