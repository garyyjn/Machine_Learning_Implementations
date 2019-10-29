from data import *
import numpy as np
import mldata
from statistics import *



class Nbayes():
	def __init__(self, m, features, labels, features_info, num_buckets,):
		self.m = int(m)
		self.buckets = int(num_buckets)
		self.features = np.transpose(features)
		self.labels = labels
		for i in range(len(labels)):
			if(labels[i]=='True'):
				labels[i]=1
			else:
				labels[i]=0
		self.features_info = features_info
		self.probs = []
		#create buckets to place features into and probabilities for each
		for i in features_info:
			if i.tup[1] == 'NOMINAL':
				buckets = len(i.tup[2])
			else:
				buckets = int(self.buckets)

			probs = np.zeros((2,buckets))
			probs[:,:] = 1 / buckets
			self.probs.append(probs)

		newX = np.copy(self.features)

		self.max = len(features_info) * [0]
		self.min = len(features_info) * [0]
		for i in range(len(features_info)):
			if features_info[i].tup[1] == 'NOMINAL':
				#standardize values
				newX[i] = self.standardizeNominal(self.features[i], i)
			else:
				#standardize values
				newX[i] = self.standardizeContinuous(self.features[i], i)

		self.features = newX

		labelCount = np.bincount(self.labels.astype(int))
		for i in range(len(self.probs)):
			smoothing = self.m*self.probs[i]
			for j in range(len(labelCount)):
				matchingFeatures = (self.features.transpose()[self.labels.astype(int) == j]).transpose()
				attributeCount = np.bincount(matchingFeatures[i].astype(float).astype(int), minlength=smoothing.shape[1])
				self.probs[i][j] = smoothing[j]+attributeCount
				self.probs[i][j] = self.probs[i][j]/labelCount[j]



	def standardizeNominal(self, x, index):
		domain = np.array(self.features_info[index].tup[2])
		sorted = np.argsort(domain)
		temp = np.searchsorted(domain, x, sorter=sorted)
		return sorted[temp]

	def standardizeContinuous(self, x, index):
		if not self.max[index]:
			self.max[index] = x.astype(float).max()
		if not self.min[index]:
			self.min[index] = x.astype(float).min()
		feature = np.copy(x.astype(float)) - self.min[index]
		feature = np.floor( feature / (self.max[index]-self.min[index])*self.buckets )
		feature[feature<0] =0
		feature[feature>=self.buckets] = self.buckets-1
		return feature

	def returnLabels(self, examples, examples_info):
		newX = examples.transpose().copy()
		for i in range(len(examples_info)):
			if examples_info[i].tup[1] == 'NOMINAL':
				#standardize values
				newX[i] = self.standardizeNominal(newX[i], i)
			else:
				#standardize values
				newX[i] = self.standardizeContinuous(newX[i], i)
		examples = newX
		labelProbs = np.bincount(self.labels.astype(int))/len(self.labels)
		results = np.zeros((2, examples.shape[1]))
		for i in range(len(labelProbs)):
			for j in range(len(self.probs)):
				attributeProbs = self.probs[j][i]
				results[i] = results[i] + np.log(attributeProbs[examples[j].astype(float).astype(int)])
			results[i] = results[i] + np.log(labelProbs[i])
		conditionalProbs = np.exp(results)
		temp= conditionalProbs[1]/(conditionalProbs[0]+conditionalProbs[1])
		for i in range(len(temp)):
			temp[i] = temp[i].round()
		return temp


def output(accuracy, accu_std, precision, precision_std, recall, recall_std, area_u_ROC):
	print("Accuracy: {:.3f} {:.3f}".format(accuracy,accu_std))
	print("Precision: {:.3f} {:.3f}".format(precision, precision_std))
	print("Recall: {:.3f} {:.3f}".format(recall, recall_std))
	print("Area under ROC: {:.3f}".format(area_u_ROC))


	return None


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


dataset = input("Dataset options: volcanoes, spam, voting: ")
crossvalidation = input("Cross validation? 0/1: ")
m = input("input value for m: ")
num_buckets = input("input value for number of buckets: ")
'''
dataset = "volcanoes"
crossvalidation = "0"
depthlimit = "0"
GR = "0"
m = "10"
num_buckets = "10"
'''
# features = None
# labels = None
# globaltype = None
# cv1 = []
# cv2 = []
# features = []
# labels = []
# test_features = []
# test_labels = []
#
# #read data and get "features_infO" which contains nominal/continuous and domain of the feature
# if dataset == "volcanoes":
# 	temp =  mldata.parse_c45("volcanoes","DATA/volcanoes")
# 	features_info = temp.schema.features[1:]
# 	features = np.array(temp)[:,1:-1]
# 	labels = np.array(temp)[:, -1]
# if dataset == "spam":
# 	temp = mldata.parse_c45("spam", "DATA/spam")
# 	features_info = temp.schema.features[1:]
# 	features = np.array(temp)[:,1:-1]
# 	labels = np.array(temp)[:, -1]
# if dataset == "voting":
# 	temp = mldata.parse_c45("voting", "DATA/voting")
# 	features_info = temp.schema.features[1:]
# 	features = np.array(temp)[:,1:-1]
# 	labels = np.array(temp)[:, -1]
# if crossvalidation == '1':
# 	None
# else:
# 	if dataset == "volcanoes":
# 		cvsets,globaltype = foldedCVVolcano()
# 		cv1 = cvsets[0]
# 		cv2 = cvsets[1]
# 		features = cv1[:, 2:-1]
# 		labels = cv1[:, -1]
# 		test_features = cv2[:, 2:-1]
# 		test_labels = cv2[:, -1]
# 	if dataset == "spam":
# 		cvsets,globaltype = foldedCVVSpam()
# 		cv1 = cvsets[0]
# 		cv2 = cvsets[1]
# 		features = cv1[:, 2:-1]
# 		labels = cv1[:, -1]
# 		test_features = cv2[:, 2:-1]
# 		test_labels = cv2[:, -1]
# 	if dataset == "voting":
# 		cvsets,globaltype = foldedCVVoting()
# 		for i in range(5):
# 			cv1[i] = cvsets[i].tuple(0)
# 			cv2[i] = cvsets[i].tuple(1)
# 			features[i] = cv1[i][:, 2:-1]
# 			labels[i] = cv1[i][:, -1]
# 			test_features[i] = cv2[i][:, 2:-1]
# 			test_labels[i] = cv2[i][:, -1]
#
# 		cv1 = cvsets[0]
# 		cv2 = cvsets[1]
# 		features = cv1[:, 2:-1]
# 		labels = cv1[:, -1]
# 		test_features = cv2[:, 2:-1]
# 		test_labels = cv2[:, -1]
# 	else:
# 		print("unsupported dataset")

data = None
if dataset == "volcanoes":
	data = read_data('..\\data\\volcanoes')
if dataset == "spam":
	data = read_data('..\\data\\spam')
if dataset == "voting":
	data = read_data('..\\data\\voting')

x, y, schema = c45_to_xy(data)
sets = foldedCV(x, y, folds=5)
accuracies = [0] * 5
precisions = [0] * 5
recalls = [0] * 5
area_under_curves = [0] * 5
best_tree = None
best_accuracy = -1;
for i in range(5):
	train_set, test_set = sets[i]
	train_x = [None] * len(train_set)
	train_y = [None] * len(train_set)
	test_x = [None] * len(test_set)
	test_y = [None] * len(test_set)
	#print(len(train_set))
	#print(len(test_set))

	for j in range(len(train_set)):
		#print(train_set[j])
		train_x[j], train_y[j] = train_set[j]

	for j in range(len(test_set)):
		#print(test_set[j])
		test_x[j], test_y[j] = test_set[j]

	n = Nbayes(m, np.array(train_x), np.array(train_y), schema, num_buckets)
	predicted_labels = n.returnLabels(np.array(test_x), schema)

	accuracies[i] = accuracy(test_y, predicted_labels)
	precisions[i] = precision(test_y, predicted_labels)
	recalls[i] = recall(test_y, predicted_labels)
	area_under_curves[i] = area_under_curve(test_y, predicted_labels)

print('Accuracy: {:.3f} {:.3f}'.format( np.mean(accuracies), np.std(accuracies)))
print('Precision:{:.3f} {:.3f}'.format( np.mean(precisions), np.std(precisions)))
print('Recall: {:.3f} {:.3f}'.format( np.mean(recalls), np.std(recalls)))
print('Area under ROC: {:.3f}'.format(np.mean(area_under_curves)))