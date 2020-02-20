import sys
import os
import csv
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials #To access authorised Spotify data


def formatted_string(string):
    return string.replace('"', "'")


# all_genres = ['pop', 'hip-hop', 'electronic', 'rock', 'country', 'folk']
all_genres = ['acoustic', 'afrobeat', 'alt-rock', 'alternative', 'ambient', 'anime', 'black-metal', 'bluegrass', 'blues', 'bossanova', 'brazil', 'breakbeat', 'british', 'cantopop', 'chicago-house', 'children', 'chill', 'classical', 'club', 'comedy', 'country', 'dance', 'dancehall', 'death-metal', 'deep-house', 'detroit-techno', 'disco', 'disney', 'drum-and-bass', 'dub', 'dubstep', 'edm', 'electro', 'electronic', 'emo', 'folk', 'forro', 'french', 'funk', 'garage', 'german', 'gospel', 'goth', 'grindcore', 'groove', 'grunge', 'guitar', 'happy', 'hard-rock', 'hardcore', 'hardstyle', 'heavy-metal', 'hip-hop', 'holidays', 'honky-tonk', 'house', 'idm', 'indian', 'indie', 'indie-pop', 'industrial', 'iranian', 'j-dance', 'j-idol', 'j-pop', 'j-rock', 'jazz', 'k-pop', 'kids', 'latin', 'latino', 'malay', 'mandopop', 'metal', 'metal-misc', 'metalcore', 'minimal-techno', 'movies', 'mpb', 'new-age', 'new-release', 'opera', 'pagode', 'party', 'philippines-opm', 'piano', 'pop', 'pop-film', 'post-dubstep', 'power-pop', 'progressive-house', 'psych-rock', 'punk', 'punk-rock', 'r-n-b', 'rainy-day', 'reggae', 'reggaeton', 'road-trip', 'rock', 'rock-n-roll', 'rockabilly', 'romance', 'sad', 'salsa', 'samba', 'sertanejo', 'show-tunes', 'singer-songwriter', 'ska', 'sleep', 'songwriter', 'soul', 'soundtracks', 'spanish', 'study', 'summer', 'swedish', 'synth-pop', 'tango', 'techno', 'trance', 'trip-hop', 'turkish', 'work-out', 'world-music']

genre = sys.argv[1]
if genre not in all_genres:
    print('Genre must be from:', all_genres)
    sys.exit(1)
file_name = 'samples/' + genre + '_sample.csv'

client_id = '' # Spotify client ID here
client_secret = '' # Spotify client secret here
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager) #spotify object to access API

unique_songs = set()

# Add any previously scraped songs to unique song set
if os.path.exists(file_name):
    previous_song_df = pd.read_csv(file_name, delimiter=',', header=0, quotechar='"')
    print('Reading previous songs...')
    for row in previous_song_df.itertuples():
        unique_songs.add((getattr(row, 'Title'), getattr(row, 'Artist')))

data_file = open(file_name, 'a+', encoding='utf-8')
writer = csv.writer(data_file, lineterminator='\n')

recommendations = sp.recommendations(seed_genres=[genre], limit=100)

songs_sample = recommendations['tracks']
for song in songs_sample:
    formatted_song_name = formatted_string(song['name'])
    formatted_song_artist = formatted_string(song['artists'][0]['name'])
    name_artist = (formatted_song_name, formatted_song_artist)
    if name_artist in unique_songs:
        continue
    unique_songs.add(name_artist)
    formatted_song_album = formatted_string(song['album']['name'])
    song_album_id = song['album']['external_urls']['spotify'].split('/')[-1]
    song_number = song['track_number']
    song_info =[formatted_song_name, formatted_song_artist, formatted_song_album, song_album_id, song_number]
    writer.writerow(song_info)
