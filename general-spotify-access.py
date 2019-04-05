import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

client_credentials_manager = SpotifyClientCredentials(client_id='2fd6c37e0dc44ebbbf7275b7a24ce182', client_secret='57b21bdf2a9c4a888a5a75661c1804a2')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
results = sp.audio_features('spotify:track:5KczWTLXxeUuy8U5uKkxeM')[0]['valence'] +sp.audio_features('spotify:track:5KczWTLXxeUuy8U5uKkxeM')[0]['tempo']
print(results)