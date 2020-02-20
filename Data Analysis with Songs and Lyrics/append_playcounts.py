import sys
import os
import csv
import time
import pandas as pd
import requests


def formatted_string(string):
    return string.replace('"', "'")


base_url = 'https://t4ils.dev:4433/api/beta/albumPlayCount?albumid='

genre = sys.argv[1]
input_file_name = 'samples/' + genre + '_sample.csv'
output_file_name = 'samples/' + genre + '_sample_counts.csv'

if not os.path.exists(input_file_name):
    print('Data for genre', genre, 'not found')
    sys.exit(1)

input_data = pd.read_csv(input_file_name, delimiter=',', header=0, quotechar='"')
output_data = open(output_file_name, 'a+', encoding='utf-8')
writer = csv.writer(output_data, lineterminator='\n')

start_index = int(input("From which song index to start scraping? "))
stop_index = int(input("At which song index to stop scraping? "))

print(input_data.head())

for idx, song in enumerate(input_data.itertuples()):
    if idx < start_index:
        continue
    elif idx > stop_index:
        break
    song_name = getattr(song, 'Title')
    song_artist = getattr(song, 'Artist')
    print(idx, ':', song_artist, '-', song_name)
    track_index = getattr(song, 'Number') - 1

    url = base_url + getattr(song, 'ID')
    response = requests.get(url, timeout=25)
    response.raise_for_status()
    album_json = response.json()
    count = album_json['data'][track_index]['playcount']

    song_info = [song_name, song_artist, count]
    writer.writerow(song_info)
    time.sleep(5)
