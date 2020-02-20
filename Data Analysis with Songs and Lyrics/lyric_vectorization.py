import numpy as np
import pandas as pd

def song2vec(lyric, wordtable):
	'''
	:param lyric: a long string with space seperated words
	:param wordtable: a list of possibly appeared words
	:return: a song by vocab space matrix
	'''
	song_vector = np.zeros((wordtable.size),dtype=int)
	song_word_as_list = lyric.split()
	for i,word in zip(range(wordtable.size),wordtable):
		song_vector[i] = song_word_as_list.count(word)
	return song_vector

def test():
	test_lyric = "The legends and the myths Achilles and his gold Hercules and his gifts Spiderman's control And Batman with his fists And clearly I don't see myself upon that list"
	wordlist = np.array(['gifts', 'myths', 'I', 'The'])
	print(song2vec(test_lyric,wordlist))

def vectorization(input, output_path, wordlist):
	songs = pd.read_csv(input)
	songs = songs.to_numpy()
	sample_count = songs.shape[0]
	origional_feature_count = songs.shape[1]
	word_target_count = wordlist.size
	feature_vectors = np.zeros((sample_count,wordlist.size))
	for i in range(sample_count):
		currentlyric = songs[i,3]
		print(currentlyric)
		currentvector = song2vec(currentlyric,wordlist)
		print(currentvector)
		feature_vectors[i] = currentvector
	print(sample_count,origional_feature_count+word_target_count)
	output = np.zeros((sample_count,origional_feature_count+word_target_count),dtype=np.object_)
	print(songs.shape)
	print(output.shape)
	output[:,0:4] = songs
	output[:,4:] = feature_vectors
	output = pd.DataFrame(output)
	output.to_csv(output_path, index=False)

word_list = pd.read_csv('filtered_wordlist')
word_list = word_list.to_numpy()
word_list = word_list[:,0]
vectorization('samples/rock_sample_lyrics.csv','rock_vectorized.csv',word_list)

