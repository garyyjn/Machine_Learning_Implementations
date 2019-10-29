import pandas as pd
import numpy as np
import random
import math
import mldata

#train_set excludes the chip index and the image ID and returns sperate data and labe. Small_set returns the first 200 only
def loadVolcanesData(train_set = False,small_set = False):
	#1. Chip index
	  #2. Image ID (see `images` folder for some examples, or UCI repo for all images)
	 # 3. Chip pixel value (8-bit unsigned integer)
	#228. Class label
	data = pd.read_csv("DATA/volcanoes/volcanoes.data")
	data = data.to_numpy()
	type = np.chararray((1, 225))
	if small_set is True:
		data = data[1:100,:]
	type.fill('C')
	if not train_set:
		return data
	else:
		features = data[:,2:227]
		labels = data[:,-1]
		return features, labels, type


def loadVotingData(train_set = False, small_set = False):
	# 1. Chip index
	# 2. Image ID (see `images` folder for some examples, or UCI repo for all images)
	# 3. Chip pixel value (8-bit unsigned integer)
	# 228. Class label
	data = pd.read_csv("DATA/voting/voting.data")
	data = data.to_numpy()
	type = np.chararray((1, data.shape[1]-1))
	if small_set is True:
		data = data[1:100, :]
	type.fill('D')
	# print(type.shape)
	if not train_set:
		return data
	else:
		features = data[:, 2:-1]
		labels = data[:, -1]
		return features, labels, type


def loadSpamData(train_set = False, small_set = False):
	data = pd.read_csv("DATA/spam/spam.data")
	data = data.to_numpy()
	type = np.chararray((1, data.shape[1] - 1))
	if small_set is True:
		data = data[1:1000
		, :]
	type.fill('C')
	type[0,4] = 'D'
	# print(type.shape)
	if not train_set:
		return data
	else:
		features = data[:, 2:-1]
		labels = data[:, -1]
		return features, labels, type


def foldedCV(data, folds = 5):
	#random.seed(2)
	sampleNumber = data.shape[0]
	fold_length = math.floor(sampleNumber/folds)
	np.random.shuffle(data)
	validationSets= [data[i*fold_length:(i+1)*fold_length,:] for i in range(folds)]
	return validationSets


def foldedCVVolcano():
	data = pd.read_csv("DATA/volcanoes/volcanoes.data")
	data = data.to_numpy()
	type = np.chararray((1, 225))
	type.fill('c')
	return foldedCV(data), type


def foldedCVVSpam():
	data = pd.read_csv("DATA/spam/spam.data")
	data = data.to_numpy()
	type = np.chararray((1, data.shape[1] - 1))
	type.fill('C')
	type[0, 4] = 'D'
	return foldedCV(data), type


def foldedCVVoting():
	data = pd.read_csv("DATA/voting/voting.data")
	data = data.to_numpy()
	print(data)
	features = data[:,0:-1]
	labels = data[:,-1]
	type = np.chararray((1, data.shape[1] - 1))
	type.fill('D')
	return foldedCV(features, labels), type


def foldedCV(data, labels, folds = 5):
	random.seed(12345)
	data_tuples = [None]*len(data)
	for i in range(len(labels)):
		data_tuples[i] = (data[i], labels[i])

	random.shuffle(data_tuples)
	fold_sets = np.array_split(np.array(data_tuples), folds)
	sets = [None] * folds

	for i in range(folds):
		print('fold ' + str(i))
		train_data = fold_sets[:i] + fold_sets[i+1:]
		train_data = np.concatenate(train_data)

		print(len(train_data))
		print(len(fold_sets[i]))
		tuple_train_test = (train_data, fold_sets[i])
		sets[i] = tuple_train_test
		# print(len(sets))
	return sets


def transform_nominal_attributes(features, schema):
	newX = features.copy()
	attribute_map = {}
	# num_features =
	for i in range(len(schema)):
		nominal_map = {}
		attribute_map[schema[i]] = nominal_map
		if schema[i].tup[1] == 'NOMINAL':
			number_value = 1
			for symbol in schema[i].tup[2]:
				nominal_map[symbol] = number_value
				number_value += 1

	for data_vector in newX:
		for i in range(len(schema)):
			if schema[i].tup[1] == 'NOMINAL':
				current_nominal_map = attribute_map[schema[i]]
				data_vector[i] = current_nominal_map[data_vector[i]]

	return newX.astype(np.float)


# Read data from a path
def read_data(path):
	pathArray = path.split('\\')
	fileName = pathArray[len(pathArray)-1]
	return mldata.parse_c45(fileName, path)


def c45_to_xy(data):
	'''
	Return np.array data and labels (x and y), x has removed index from c45
	:param data: example set
	:return: tuple(np.array)
	'''
	schema = data.schema
	data = np.array(data)
	assert len(data[:, 1:-1]) == len(data[:, -1])
	return data[:, 1:-1], data[:, -1], schema[1:-1]
