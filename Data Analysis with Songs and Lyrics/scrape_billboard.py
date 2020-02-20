import time
import random
import pandas as pd
import billboard


def reformat_string(string):
    return '"' + string.replace('"', "'") + '"'


previous_song_df = pd.read_csv('billboard.csv', delimiter=',', header=0, quotechar='"')
unique_songs = set()
data_file = open('billboard.csv', 'a')

start_date = str(input('Latest date to retrieve (yyyy-mm-dd) [blank for most recent chart]: '))

# Add previously scraped songs to unique song set
print('Reading billboard.csv...')
for row in previous_song_df.itertuples():
    unique_songs.add((getattr(row, 'Title'), getattr(row, 'Artist')))

chart = billboard.ChartData('hot-100', date=start_date)
# while chart.previousDate:
for i in range(5):
    print('Using chart ' + str(chart.date))
    for song in chart:
        formatted_song_title = reformat_string(song.title)
        formatted_song_artist = reformat_string(song.artist)
        if (formatted_song_title, formatted_song_artist) in unique_songs:
            continue
        unique_songs.add((formatted_song_title, formatted_song_artist))
        song_info = ",".join([formatted_song_title, formatted_song_artist, str(song.rank), str(song.peakPos), str(song.lastPos), str(song.weeks)])
        data_file.write(song_info + '\n')
    chart = billboard.ChartData('hot-100', chart.previousDate)
    time.sleep(10)
