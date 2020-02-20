#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import scipy
import nltk
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


data = pd.read_csv('./samples/country_sample_lyrics.csv')
# plt.scatter(data['Count'], data['Diversity'])
plt.hist(data['Count'])
# plt.xlim((0, 300))
plt.xlabel('Listens')
plt.ylabel('Count')
plt.show()


# In[4]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('vader_lexicon')
nltk.download('stopwords')

SID = SentimentIntensityAnalyzer()
data['Polarity'] = data['Lyrics'].apply(lambda x: SID.polarity_scores(x)['compound'])

plt.scatter(data['Count'], data['Polarity'])
plt.xlabel('Listen Count')
plt.ylabel('Polarity of lyrics')
plt.yticks([-1, 0, 1], ['Negative', 'Neutral', 'Positive'])


# In[5]:


# filtering stop words
import re
from nltk.corpus import stopwords
stopwords_eng = stopwords.words('english')

data['Lyrics'] = data['Lyrics'].apply(lambda x : x.lower())
data['Lyrics'] = data['Lyrics'].apply(lambda x : re.sub(r'\d+','', x))
data['Lyrics'] = data['Lyrics'].apply(lambda x : ' '.join([word for word in re.split(r'[\W_]' ,x) if word not in stopwords_eng]))
data['Lyrics'] = data['Lyrics'].apply(lambda x : re.sub(r"i'm",'', x))
data['Lyrics'] = data['Lyrics'].apply(lambda x : re.sub(r"you",'', x))
data['Lyrics'] = data['Lyrics'].apply(lambda x : re.sub(r"yeah",'', x))
data['Lyrics'] = data['Lyrics'].apply(lambda x : re.sub(r"oh",'', x))


# In[6]:


romantic_words = set([
    'love', 'lover', 'baby','luv', 'beautiful', 'cute', 'loving', 'gorgeous', 'fine', 'darling'
])

data['Romantic words'] = data['Lyrics'].apply(lambda x : len([word for word in re.split(r'[\W_]' ,x) if word in romantic_words]))
# plt.scatter(data['Count'], data['Romantic words'])

data['Love existence'] = data['Lyrics'].apply(lambda x : len([word for word in re.split(r'[\W_]' ,x) if word in romantic_words]) > 0)

# plt.scatter(data['Count'], data['Romantic words'], c=data['Love existence'])

data.sort_values(by='Count')

bins = [0, 5, 10, 20, 40]
labels = ['Cat{}'.format(x) for x in bins[1::]]
data['Romance category'] = pd.cut(data['Romantic words'], bins=bins, labels=labels)
data.head()  
data[data['Romantic words'] > 10]['Romantic words'].plot(kind='hist')
plt.xlabel('Count of romantic words used')


# In[7]:


plt.scatter(data['Count'], data['Romantic words'])
plt.xlabel('Play count')
plt.ylabel('Love count')


# In[8]:


cuss_words = set([
    'fuck', 'shit', 'bitch', 'nigga', "fuckin'"
])

data['Cuss words'] = data['Lyrics'].apply(lambda x : len([word for word in re.split(r'[\W_]' ,x) if word in cuss_words]))
# plt.scatter(data['Count'], data['Romantic words'])

data['Cuss existence'] = data['Lyrics'].apply(lambda x : len([word for word in re.split(r'[\W_]' ,x) if word in cuss_words]) > 0)

# plt.scatter(data['Count'], data['Romantic words'], c=data['Love existence'])

data.sort_values(by='Count')

bins = [0, 5, 10, 20, 40]
labels = ['Cat{}'.format(x) for x in bins[1::]]
data['Cuss category'] = pd.cut(data['Cuss words'], bins=bins, labels=labels)
data.head()  
plt.scatter(data['Count'], data['Cuss words'])
plt.xlabel('Count of cuss words used')


# In[9]:


# Finding diversity score of lyrics with correlation of popularity
def lexical_diversity(my_text_data):
    word_count = len(my_text_data)
    vocab_size = len(set(my_text_data))
    diversity_score = word_count / vocab_size
    return diversity_score

data['Diversity'] = data['Lyrics'].apply(lambda x : lexical_diversity(x))

# plt.scatter(data['Count'], data['Diversity'])
plt.hist(data['Diversity'], bins=100)
# plt.xlim((0, 300))
plt.xlabel('Word diversity')
plt.ylabel('Count')
plt.show()


# In[10]:


from nltk.cluster.kmeans import KMeansClusterer
from sklearn.cluster import KMeans

cls = KMeans(2)
# cls.fit(voting)
sub_data = np.asmatrix(data[['Cuss words', 'Romantic words', 'Count']], dtype=np.float)
cls.fit(sub_data)
clusters = cls.predict(sub_data)
# transformed_centers = pca.transform(cls.means())
centers = cls.cluster_centers_

plt.scatter(data['Cuss words'], data['Count'], cmap=plt.cm.Spectral, c=clusters)
plt.scatter(centers[:, 0], centers[:, -1], c=np.random.rand(len(centers),), s=100, alpha=1)
plt.title('Play count vs Cuss words with cluster_centers')
plt.xlabel('Cuss words count')
plt.ylabel('Play count')
plt.show()


# In[11]:


plt.scatter(data['Romantic words'], data['Count'], cmap=plt.cm.Spectral, c=clusters)
plt.scatter(centers[:, 1], centers[:, -1], c=np.random.rand(len(centers),), s=100, alpha=1)
plt.title('Play count vs Romantic words with cluster_centers')
plt.xlabel('Romantic words count')
plt.ylabel('Play count')
plt.show()


# Okay, let's try making this a categorical problem, say anything with more than 200 million plays is very popular

# In[15]:


data['Popular'] = data['Count'].apply(lambda x: 1 if x > 200000000 else 0)

plt.scatter(data['Romantic words'], data['Cuss words'], c=data['Popular'])

