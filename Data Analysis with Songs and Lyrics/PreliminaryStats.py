import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv
import io
from nltk.corpus import stopwords
import re
import wordcloud

stop = stopwords.words('english')


data = pd.read_csv("billboard_w_lyrics.csv")
data['Lyrics'] = data['Lyrics'].apply(lambda x : x.lower())
data['Lyrics'] = data['Lyrics'].apply(lambda x : re.sub(r'\d+','', x))
data['Lyrics'] = data['Lyrics'].apply(lambda x : ' '.join([word for word in x.split() if word not in (stop)]))
data['Lyrics'] = data['Lyrics'].apply(lambda x : re.sub(r"i'm",'', x))
data['Lyrics'] = data['Lyrics'].apply(lambda x : re.sub(r"you",'', x))

data.to_csv(r'billboard_formatted_lyrics.csv')

data = data.to_numpy()
print(data.shape)
peak = data[:,3]
lyrics = data[:,-1]
def stdavg(peak):
    peak_std = np.std(peak)
    peak_avg = np.sum(peak)/peak.size
    print(peak_std,peak_avg)

appearedwords = []
wordrepeats = []

concatenatedString = ""

for i in range(lyrics.size):
    concatenatedString += " "
    concatenatedString += lyrics[i]

table = str.maketrans(dict.fromkeys(string.punctuation))  # OR {key: None for key in string.punctuation}
concatenatedString = concatenatedString.translate(table)  # Output: string without punctuation
words_from_all = concatenatedString.split()
words_from_all = np.asarray(words_from_all)

print(words_from_all)
print(words_from_all.shape)

word,count = np.unique(words_from_all,return_counts=True)
count, words = zip(*sorted(zip(count, word),reverse=True))
with io.open("formatted_count.csv", "w", encoding="utf-8",newline='') as f:
    writer = csv.writer(f)
    writer.writerows(zip(words,count))
x_count_plot = range(len(count))
plt.bar(x=x_count_plot[:1000],height=count[:1000])
plt.yscale('log')
plt.xlabel("Ranking of word apparence")
plt.ylabel("Apparence of word")
plt.title("Apparence of words in 500 Billboard Songs")
plt.show()