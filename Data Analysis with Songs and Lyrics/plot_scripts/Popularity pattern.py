#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import scipy
import numpy as np
import matplotlib.pyplot as plt


# In[37]:


data = pd.read_csv('./samples/country_sample_lyrics.csv')
# plt.scatter(data['Count'], data['Diversity'])
plt.hist(data['Count'], bins=100)
# plt.xlim((0, 300))
plt.xlabel('Listens')
plt.ylabel('Count')
plt.show()


# In[38]:


data.describe()


# Our minimum and maximum is 9700000 and 1750000000 respectively. So, we can see if there is any patterns for more than a certain amount of play

# In[39]:


from sklearn.feature_extraction.text import CountVectorizer

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
data['Lyrics'] = data['Lyrics'].apply(lambda x : re.sub(r"oh+",'', x))
data['Lyrics'] = data['Lyrics'].apply(lambda x : re.sub(r"like",'', x))


# In[40]:


def extract_features(df):
    vectorizer = CountVectorizer()
    results = vectorizer.fit_transform(df)
    print(vectorizer.get_feature_names())
    return results, vectorizer

counts, vectorizer = extract_features(data['Lyrics'])


# In[41]:


# Plot most common features
def plot_most_common(counts, vectorizer, number_words):
    import seaborn as sns
    words = vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in counts:
        total_counts+=t.toarray()[0]
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:number_words]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 

    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='{} most common words'.format(number_words))
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()
    
plot_most_common(counts, vectorizer, 20)


# In[42]:


from sklearn.decomposition import LatentDirichletAllocation as LDA

 
# Helper function
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        
number_topics = 2
number_words = 20# Create and fit the LDA model
lda = LDA(n_components=number_topics, n_jobs=-1)
lda.fit(counts)# Print the topics found by the LDA model
print("Topics found via LDA:")
print_topics(lda, vectorizer, number_words)


# In [*]

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(distance_threshold=0.0, n_clusters=None)

model = model.fit(counts.todense())

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
