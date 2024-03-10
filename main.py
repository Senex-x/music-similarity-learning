import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

if __name__ == '__main__':

    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
    sp.trace = False

    # find album by name
    album = "Pestbringer"
    results = sp.search(q="album:" + album, type="album")

    # get the first album uri
    album_id = results['albums']['items'][0]['uri']

    # get album tracks
    tracks = sp.album_tracks(album_id)
    for track in tracks['items']:
        print(track['name'])

# SPOTIPY_CLIENT_ID='0af36c3b046a4f9db32db4daab40dae6'
# SPOTIPY_CLIENT_SECRET='9bb0f527ffe045ca9c3d04344f121b06'
