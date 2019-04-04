import spotipy
import spotipy.util
import sys
#from json.decoder import JSONDecodeError
#martinjchu

username = sys.argv[1]
scope = 'user-library-read'
token = spotipy.util.prompt_for_user_token(username,scope,client_id='2fd6c37e0dc44ebbbf7275b7a24ce182',client_secret='57b21bdf2a9c4a888a5a75661c1804a2', redirect_uri= 'https://google.com')
spotify = spotipy.Spotify(auth=token)
#results = spotify.search(q='artist:' + "Drake", type='artist')
results = spotify.audio_features('spotify:track:7xYnUQigPoIDAMPVK79NEq')

print results