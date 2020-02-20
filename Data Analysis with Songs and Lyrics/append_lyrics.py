import sys
import os
import csv
import lyricsgenius
import pandas as pd
import re
import string


# Return song title without featuring artists (Genius search does not work well with these)
def searchable_title(title):
    return re.sub(r'\ - (.*)| \(?feat\. (.*)| \(?featuring (.*)| \(with (.*)| \(& (.*)| \(and (.*)|', '', title)


# Return string without bracketed annotations, newlines, or double quotes
def format_string(string):
    return re.sub(r'\[(.*?)\]', ' ', string).replace('\n', ' ').replace('\r', ' ').replace('"', "'")


genius = lyricsgenius.Genius('') #insert API token here
genius.verbose = False
genius.remove_section_headers = True
genius.skip_non_songs = True

genre = sys.argv[1]
input_file_name = 'samples/' + genre + '_sample_counts.csv'
output_file_name = 'samples/' + genre + '_sample_lyrics.csv'

if not os.path.exists(input_file_name):
    print('File not found')
    sys.exit(1)

input_data = pd.read_csv(input_file_name, delimiter=',', header=0, quotechar='"')
output_data_file = open(output_file_name, 'a+', encoding='utf-8')
writer = csv.writer(output_data_file, lineterminator='\n')

start_index = int(input("From which song index to start scraping? "))
stop_index = int(input("At which song index to stop scraping? "))

print(input_data.head())

for idx, row in enumerate(input_data.itertuples()):
    if idx < start_index:
        continue
    elif idx > stop_index:
        break
    title = getattr(row, 'Title')
    artist = getattr(row, 'Artist')
    count = getattr(row, 'Count')
    print(idx, ":", artist, "-", title)
    song = genius.search_song(searchable_title(title), artist)
    if song is None:
        continue
    formatted_lyrics = format_string(song.lyrics)
    printable_lyrics = ''.join(filter(lambda x: x in string.printable, formatted_lyrics))
    writer.writerow([title, artist, count, printable_lyrics])
