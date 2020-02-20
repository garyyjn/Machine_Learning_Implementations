import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso, Ridge
from sklearn.preprocessing import Normalizer
#title, artist, count, lyric, vectorization 21534
def load_data_popularity_listed():
	freq_vectors = []
	popularity = []
	hip_pop = pd.read_csv('vectorizations/hip-hop_vectorized.csv')
	pop = pd.read_csv('vectorizations/pop_vectorized.csv')
	print(pop.head())
	rock = pd.read_csv('vectorizations/rock_vectorized.csv')
	country = pd.read_csv('vectorizations/vectorized_country.csv')
	folk = pd.read_csv('vectorizations/vectorized_folk.csv')
	print(hip_pop.head)
	hip_pop = hip_pop.to_numpy()
	pop = pop.to_numpy()
	rock = rock.to_numpy()
	country = country.to_numpy()
	folk = folk.to_numpy()
	word_freq_vectors = hip_pop[:,4:]
	popularity_freq_vector = hip_pop[:,2]
	freq_vectors.append(word_freq_vectors)
	popularity.append(popularity_freq_vector)
	word_freq_vectors = pop[:,4:]
	popularity_freq_vector = pop[:,2]
	freq_vectors.append(word_freq_vectors)
	popularity.append(popularity_freq_vector)
	word_freq_vectors = rock[:,4:]
	popularity_freq_vector = rock[:,2]
	freq_vectors.append(word_freq_vectors)
	popularity.append(popularity_freq_vector)
	word_freq_vectors = country[:,4:]
	popularity_freq_vector = country[:,2]
	freq_vectors.append(word_freq_vectors)
	popularity.append(popularity_freq_vector)
	word_freq_vectors = folk[:,4:]
	popularity_freq_vector = folk[:,2]
	freq_vectors.append(word_freq_vectors)
	popularity.append(popularity_freq_vector)
	return freq_vectors, popularity

def linearRegression(x,y):
	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)
	#linRegModel = LinearRegression().fit(X_train,y_train)
	linRegModel = Lasso(alpha=80000, max_iter= 1000).fit(X_train, y_train)
	train_score = linRegModel.score(X_train,y_train)
	validation_score = linRegModel.score(X_test, y_test)
	print("test\\ validation score: {} {}".format(train_score, validation_score))
	weights = linRegModel.coef_
	idx = (-np.abs(weights)).argsort()[:5]
	for i in idx:
		print(index2word(word_list,i),weights[i])

def index2word(wordlist,n):
	return wordlist[n]

#x_list, y_list = load_data_popularity_listed()
hip_pop = pd.read_csv('vectorizations/hip-hop_vectorized.csv')
hip_pop = hip_pop.to_numpy()
word_freq_vectors = hip_pop[:,4:304]
popularity_freq_vector = hip_pop[:,2].astype(np.float)
normalizer = Normalizer()
'''
popularity_avg = np.average(popularity_freq_vector)
popularity_std = np.std(popularity_freq_vector)
popularity_freq_vector = (popularity_freq_vector-popularity_avg)/(popularity_std**.5)
'''
word_list = pd.read_csv('filtered_wordlist')
word_list = word_list.to_numpy()
word_list = word_list[:,0]
linearRegression(word_freq_vectors, popularity_freq_vector)
